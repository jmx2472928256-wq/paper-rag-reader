# -*- coding: utf-8 -*-
"""
RAG 问答链路模块。

流程
----
    用户问题 -> 混合检索(BM25+FAISS) -> 上下文组装(带编号来源)
             -> GLM-5.3 受约束生成 -> 引用溯源 -> 结构化答案

防编造（幻觉）约束
------------------
1. System prompt 明确规定：仅可依据【文献片段】作答；证据不足时必须
   直接回答"根据当前文献库，无法回答该问题"；禁止输出文献中不存在的
   数字与结论；关键论断须以 [编号] 标注来源；
2. 低温采样（默认 temperature=0.1），抑制模型的自由发挥；
3. 生成后做机械校验：解析答案中的 [n] 引用标记，映射回检索命中的
   chunk（论文-章节-页码），并标出"引用了不存在的编号"的异常标记，
   供 evaluator 的幻觉率检测使用。

输出结构（ask 返回值）
----------------------
{
  "question":  "...",
  "answer":    "模型回答原文（含 [n] 标记）",
  "insufficient": false,          # 是否触发"无法回答"话术
  "citations": [                  # 引用溯源结果
      {"marker": 1, "paper_title": "...", "section": "method",
       "page": 4, "chunk_id": "...", "quote": "..."}, ...
  ],
  "invalid_markers": [7],         # 答案中引用了不存在编号（幻觉信号）
  "balanced": true,               # 是否启用按文献均衡召回（全库/多source提问时）
  "per_source_counts": {"paper1.pdf": 5, "paper5.pdf": 2},  # 每篇入选块数
  "params": {"top_k": 12, "mode": "hybrid", "rerank": true,
             "candidates": 18, "temperature": 0.1, "source": "全部(均衡召回)"},
  "unrecalled_sources": [         # 未召回文档（显式source过滤或全库模式）
      {"source": "paper5.pdf", "reason": "not_recalled"}  # 或 not_in_index / no_relevant
  ],
  "retrieved": [                  # 送入上下文的检索命中
      {"rank": 1, "score": 0.83, "rerank_score": 0.61, "chunk_id": "...",
       "source": "...", "section": "method", "page": 4, "text": "..."}, ...
  ]
}

用法
----
    python -m src.rag_pipeline "什么是XXX？"    # 单问
    python -m src.rag_pipeline                  # 进入交互问答
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path

try:
    from . import retrieval
    from .config import (
        QA_LOG_FILE, QA_TEXT_LOG, RETRIEVE_LOG, RERANK_CANDIDATES,
        RERANK_ENABLED, chat, clamp_top_k, ensure_dirs, setup_console,
    )
    from .reranker import LightweightReranker
except ImportError:  # 以脚本方式直接运行
    import retrieval
    from config import (
        QA_LOG_FILE, QA_TEXT_LOG, RETRIEVE_LOG, RERANK_CANDIDATES,
        RERANK_ENABLED, chat, clamp_top_k, ensure_dirs, setup_console,
    )
    from reranker import LightweightReranker

# ==========================================================================
# Prompt 模板
# ==========================================================================
# System prompt：以硬性规则约束模型只依据检索证据作答，是防幻觉的第一道闸门。
# v3（加固版）：显式给出"先查证、后作答"的决策流程，把防幻觉从抽象规则
# 变成可执行的判断步骤。
SYSTEM_PROMPT = """你是一个严谨的学术文献研读助手。你唯一的信息来源是本次对话中提供的【文献片段】，除此之外你没有任何关于这些论文的可靠信息。

【第一步 · 证据检查（回答前必须执行）】
通读全部文献片段，判断其中是否包含回答问题所需的信息：
- 包含足够信息 → 按【第二步 · 作答规则】组织回答；
- 不包含或不足以回答 → 必须直接回答："根据当前文献库，未检索到相关资料，无法回答该问题"，并简要说明已查找的文献范围；此情况下严禁输出任何推测、外部知识或"可能"的答案。

【第二步 · 作答规则】
1. 【唯一信息来源】只使用【文献片段】中的信息作答；禁止引入外部知识、训练记忆、常识推测或个人观点；
2. 【禁止编造】严禁虚构文献片段中不存在的章节、方法、数据、结论或文献来源；宁可少答，不可编造；
3. 【数值保真】数字、百分比、指标、年份必须与片段原文完全一致，禁止四舍五入、估算或改写；
4. 【引用溯源】每个关键论断末尾用 [编号] 标注来源（编号=片段序号），涉及具体论文时写明来源文档名称（片段头部"来源文件:"后的文件名）与所在章节；
5. 【逐篇覆盖】当片段来自多篇文档时，分别说明每篇文档与问题的关系，不得遗漏任何一篇；
6. 【如实存疑】文献中表述模糊、不完整或相互矛盾时，明确指出不确定性；宁可不答，不可错答；
7. 【禁止内部标记】严禁输出任何内部调试标记、占位符或系统提示，如 [Advisor consultation #1]、[Advisor review]、[End of advisor consultation #1] 等；你的回答必须直接是答案内容本身，不能包含任何方括号内的系统/调试/占位文本（引用标记 [n] 除外）；
8. 用简洁的学术中文回答，可用要点列表组织内容。"""

USER_PROMPT_TEMPLATE = """【文献片段】
{context}

【问题】
{question}

请严格按照系统规则作答。"""

# 答案中"无法回答"话术识别（机械校验用）
_INSUFFICIENT_RE = re.compile(
    r"无法回答|未能?找到|没有(找到|足够|相关)|信息不足|不足以回答"
    r"|insufficient|cannot\s+answer",
    re.IGNORECASE,
)
# 引用标记: [1] / [12]
_CITATION_RE = re.compile(r"\[(\d{1,2})\]")


def section_zh(name: str) -> str:
    """规范章节名的中文展示名。"""
    return {
        "abstract": "摘要", "introduction": "引言", "related_work": "相关工作",
        "method": "方法", "experiment": "实验", "results": "结果",
        "discussion": "讨论", "conclusion": "结论", "references": "参考文献",
        "other": "其他",
    }.get(name, name)


# 创新点相关 query 关键词（用于检索增强与 prompt 分支）
_INNOVATION_KEYWORDS = {"创新点", "创新", "研究贡献", "研究创新", "贡献", "novelty", "contribution", "innovative"}


def _is_innovation_query(question: str) -> bool:
    """判断是否为创新点/研究贡献类提问。"""
    q = (question or "").lower()
    return any(kw in q for kw in _INNOVATION_KEYWORDS)


class RAGPipeline:
    """端到端 RAG 问答器：检索 -> 受约束生成 -> 引用溯源。"""

    def __init__(self, retriever=None, top_k: int = 12, mode: str = "hybrid",
                 temperature: float = 0.1, exclude_references: bool = True):
        """
        Args:
            retriever: HybridRetriever 实例；为 None 时自动加载/构建索引。
            top_k: 送入上下文的检索块数（默认12，缓解多文档查询时短文档
                   被长文档挤出召回）。
            mode: 检索模式（bm25 / dense / hybrid），供消融实验切换。
            temperature: 生成温度（低温防编造）。
            exclude_references: 是否把 references 章节的块排除出上下文
                （参考文献条目通常与提问无关，混入会稀释证据质量）。

        重排序: 开关取 RERANK_ENABLED（.env: RERANK_ENABLED），开启时检索
        先取 RERANK_CANDIDATES 个候选，经 LightweightReranker 重排后取
        top_k 送入大模型；ask(rerank=...) 可按次覆盖。
        """
        self.retriever = retriever or retrieval.load_or_build_retriever()
        self.top_k = top_k
        self.mode = mode
        self.temperature = temperature
        self.reranker = LightweightReranker()
        self.rerank_enabled = RERANK_ENABLED
        self.allowed_sections = None
        if exclude_references:
            self.allowed_sections = {
                s for s in ("abstract", "introduction", "related_work", "method",
                            "experiment", "results", "discussion", "conclusion", "other")
            }

    # ------------------------------------------------------------------
    def _build_context(self, hits: list):
        """
        将检索命中组装为带编号来源的上下文文本。

        每块头部明确标注：原始PDF文件名(source)、论文标题、章节、页码，
        使模型回答"内容来自哪个文件"时有据可依。
        """
        blocks = []
        for i, hit in enumerate(hits, start=1):
            c = hit["chunk"]
            header = (
                f"[{i}] 来源文件: {c.get('source') or '未知'} | "
                f"论文:《{c.get('paper_title', '未知论文')}》 "
                f"{section_zh(c.get('section', 'other'))}部分 "
                f"(约第{c.get('page', '?')}页)"
            )
            blocks.append(f"{header}\n{c['text']}")
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    def trace_citations(self, answer: str, hits: list) -> dict:
        """
        引用溯源：解析答案中的 [n] 标记并映射回检索证据。

        Returns:
            {"citations": [...], "invalid_markers": [...]}
            invalid_markers 非空说明模型引用了不存在的编号，属于幻觉信号。
        """
        # 去重保序，只溯源实际出现的标记
        seen = []
        for m in _CITATION_RE.findall(answer):
            n = int(m)
            if n not in seen:
                seen.append(n)

        citations, invalid = [], []
        for n in seen:
            if 1 <= n <= len(hits):
                c = hits[n - 1]["chunk"]
                citations.append({
                    "marker": n,
                    "source": c.get("source", ""),
                    "paper_title": c.get("paper_title", ""),
                    "section": c.get("section", ""),
                    "page": c.get("page"),
                    "chunk_id": c.get("chunk_id", ""),
                    # 截取原文片段，方便人工快速核对
                    "quote": c.get("text", "")[:160],
                })
            else:
                invalid.append(n)
        return {"citations": citations, "invalid_markers": invalid}

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def retrieve(self, question: str, top_k: int = None, mode: str = None,
                 metadata_filter=None, rerank: bool = None,
                 save_log: bool = True) -> tuple:
        """
        独立检索接口（ask 与 Web「检索」页共用同一套策略）。

        策略：未指定 source 或多篇 source -> 按文献均衡召回；单篇 -> 普通检索。
        重排开启时先取 max(top_k, RERANK_CANDIDATES) 个候选，重排后取 top_k，
        并对均衡模式做逐篇覆盖保底。

        Args:
            save_log: 是否写入检索调试日志 logs/retrieve.log（任务④：
                      每一次检索都记录 时间戳/query/参数/每块来源与分数/文本预览）。

        Returns:
            (hits, info)；info = {"src_list", "use_balance", "final_k",
            "rerank_on", "fetch_k"}，供调用方记录参数。
        """
        t0 = time.time()
        src_list = (metadata_filter or {}).get("source") if metadata_filter else None
        if isinstance(src_list, str):
            src_list = [src_list]
        use_balance = (not src_list) or len(src_list) >= 2

        # 任务⑤：top_k 数值边界保护（非法/超界输入钳制，不崩溃）
        final_k = clamp_top_k(top_k, self.top_k)
        rerank_on = self.rerank_enabled if rerank is None else bool(rerank)

        # 创新点查询增强：单文献场景下提高候选池 + 双路召回
        innovation_query = _is_innovation_query(question)
        if innovation_query and not use_balance and src_list and len(src_list) == 1:
            final_k = clamp_top_k(top_k, self.top_k) * 3
            rerank_on = True

        fetch_k = max(final_k, RERANK_CANDIDATES) if rerank_on else final_k

        if use_balance:
            hits = self.retriever.search_balanced(
                question, top_k=fetch_k, mode=mode or self.mode,
                allowed_sections=self.allowed_sections, sources=src_list)
        else:
            hits = self.retriever.search(
                question, top_k=fetch_k, mode=mode or self.mode,
                allowed_sections=self.allowed_sections,
                metadata_filter=metadata_filter)

        # 创新点查询增强：补充 BM25 关键词召回（双路检索）
        if innovation_query and not use_balance and src_list and len(src_list) == 1:
            bm25_hits = self.retriever.search(
                question, top_k=fetch_k, mode="bm25",
                allowed_sections=self.allowed_sections,
                metadata_filter=metadata_filter)
            # 合并并去重（按 chunk_id）
            seen = {h["chunk_id"] for h in hits}
            for h in bm25_hits:
                if h["chunk_id"] not in seen:
                    hits.append(h)
                    seen.add(h["chunk_id"])
            # 重新按分数排序，取前 fetch_k 个
            hits = sorted(hits, key=lambda x: x.get("score", 0), reverse=True)[:fetch_k]

        if rerank_on:
            scored = self.reranker.rerank(question, hits)  # 全量候选打分不截断
            hits = scored[:final_k]
            if use_balance:
                hits = self._ensure_source_coverage(scored, hits, final_k)
            for r, h in enumerate(hits, 1):
                h["rank"] = r

        info = {"src_list": src_list, "use_balance": use_balance,
                "final_k": final_k, "rerank_on": rerank_on, "fetch_k": fetch_k}

        # 任务⑦：关键运行日志打印到控制台，便于调试
        src_desc = src_list if src_list else ("全部文献(均衡召回)" if use_balance else "全库")
        print(f"[rag] 检索完成: 范围={src_desc} 候选={fetch_k} "
              f"重排={'开' if rerank_on else '关'} 返回={len(hits)}块 "
              f"耗时={time.time() - t0:.2f}s")
        if save_log:
            self._append_retrieve_log(question, hits, info)
        return hits, info

    def _append_retrieve_log(self, question: str, hits: list, info: dict) -> None:
        """
        检索调试日志（任务④）：logs/retrieve.log 记录时间戳、query、参数、
        每个chunk的来源/分数/文本预览——不含模型输出（问答全文见 logs/qa.log）。
        """
        ensure_dirs()
        now = datetime.now().isoformat(timespec="seconds")
        lines = ["=" * 70,
                 f"[{now}] 检索调试 query: {question}",
                 f"参数: {json.dumps(info, ensure_ascii=False, default=str)}",
                 f"命中 {len(hits)} 条:"]
        for h in hits:
            c = h.get("chunk", {})
            preview = " ".join(str(c.get("text", "")).split())[:120] or "(空文本)"
            rr = h.get("rerank_score")
            lines.append(
                f"  ({h.get('rank')}) 检索分={h.get('score')}"
                + (f" 重排分={rr}" if rr is not None else "")
                + f" | {c.get('source') or '未知文件'}"
                + f" | {c.get('section') or '其他'} p{c.get('page', '—')}"
                + f" | {c.get('chunk_id', '')}")
            lines.append(f"      文本: {preview}")
        with open(RETRIEVE_LOG, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def ask(self, question: str, top_k: int = None, mode: str = None,
            save_log: bool = True, metadata_filter=None, rerank: bool = None) -> dict:
        """
        回答一个学术问题。

        Args:
            question: 用户问题。
            top_k: 最终送入大模型的块数（重排开启时也决定候选截断）。
            mode: 检索模式（bm25 / dense / hybrid），供消融实验切换。
            save_log: 是否将问答追加到日志（output/qa_log.jsonl + qa_log.txt）。
            metadata_filter: 元数据过滤参数，透传给检索器。如
                {"source": "paper1.pdf"} 只在该 PDF 的分块内检索作答。
            rerank: 是否启用 Reranker 重排序；None 时取 RERANK_ENABLED 配置。
                开启时检索先取 RERANK_CANDIDATES 个候选重排，再取 top_k。

        Returns:
            见模块 docstring 中的输出结构说明。
        """
        # 1) 检索（独立方法，供 ask 与 Web「检索」页复用同一套策略）
        hits, rinfo = self.retrieve(question, top_k=top_k, mode=mode,
                                    metadata_filter=metadata_filter, rerank=rerank)
        src_list = rinfo["src_list"]
        use_balance = rinfo["use_balance"]
        rerank_on = rinfo["rerank_on"]
        final_k = rinfo["final_k"]
        fetch_k = rinfo["fetch_k"]

        params = {
            "top_k": final_k,
            "mode": mode or self.mode,
            "rerank": rerank_on,
            "candidates": fetch_k if rerank_on else None,
            "source": src_list if src_list else
                      ("全部(均衡召回)" if use_balance else "全库"),
            "temperature": self.temperature,
        }

        base = {
            "question": question,
            "answer": "根据当前文献库，无法回答该问题。",
            "insufficient": True,
            "citations": [],
            "invalid_markers": [],
            "unrecalled_sources": [],
            "balanced": use_balance,
            "per_source_counts": {},
            "params": params,
            "retrieved": [],
        }
        if not hits:
            # 检索为空时同样做零召回检测，方便定位是过滤条件还是语料问题
            if use_balance and not src_list:
                base["unrecalled_sources"] = self._auto_zero_recall({})
            else:
                base["unrecalled_sources"] = self._zero_recall_sources(metadata_filter, [])
            return base

        # 每篇文献的入选块数（覆盖统计，供展示与"逐篇解答"要求使用）
        per_source: dict = {}
        for h in hits:
            s = h["chunk"].get("source") or "未知"
            per_source[s] = per_source.get(s, 0) + 1

        # 2) 组装受约束 prompt 并调用 LLM
        context = self._build_context(hits)
        user_content = USER_PROMPT_TEMPLATE.format(context=context, question=question)
        # 多文献覆盖时，明确要求逐篇解答，杜绝遗漏短篇文献
        if use_balance and len(per_source) >= 2:
            covered = "、".join(per_source.keys())
            user_content += (
                f"\n\n【回答要求】本次证据覆盖以下文献: {covered}。"
                "请分别针对其中每一篇文献给出对应解答（每篇单独列出要点），"
                "与问题无关的文献请如实说明其相关性；不得遗漏任何一篇。"
            )
        # 创新点查询专项约束
        if _is_innovation_query(question):
            user_content += (
                "\n\n【创新点提取规则】"
                "1. 仅使用上述【文献片段】中的原文内容提取创新点，不得引入外部知识；"
                "2. 如果上述片段中找不到本文献的创新点，直接输出："
                "【当前检索片段未找到本文献创新点】；"
                "3. 禁止输出任何内部标记、调试信息或占位符（如 [Advisor consultation #1]）。"
            )
            # 控制台日志：打印该文献召回的全部 chunk 文本，方便调试
            print(f"[rag] 创新点查询召回全量chunk文本 (source={src_list}):")
            for h in hits:
                c = h.get("chunk", {})
                text_preview = " ".join(str(c.get("text", "")).split())[:300]
                print(f"  [{c.get('section')} p{c.get('page')}] {text_preview}")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        answer = chat(messages, temperature=self.temperature)

        # 3) 引用溯源 + 机械校验
        traced = self.trace_citations(answer, hits)
        # 任务⑦：关键运行日志打印到控制台
        print(f"[rag] 回答生成: {len(answer)}字符 | 引用{len(traced['citations'])}处 | "
              f"无效引用{len(traced['invalid_markers'])}处 | "
              f"拒答={'是' if _INSUFFICIENT_RE.search(answer) else '否'}")

        result = {
            "question": question,
            "answer": answer,
            "insufficient": bool(_INSUFFICIENT_RE.search(answer)),
            "citations": traced["citations"],
            "invalid_markers": traced["invalid_markers"],
            "balanced": use_balance,
            "per_source_counts": per_source,
            "params": params,
            "retrieved": [
                {
                    "rank": h["rank"], "score": h["score"],
                    "rerank_score": h.get("rerank_score"),
                    "chunk_id": h["chunk_id"],
                    "source": h["chunk"].get("source", ""),
                    "paper_title": h["chunk"].get("paper_title", ""),
                    "section": h["chunk"].get("section", ""),
                    "page": h["chunk"].get("page"),
                    "text": h["chunk"].get("text", ""),
                }
                for h in hits
            ],
        }
        # 零召回检测：显式 source 过滤用原逻辑；全库均衡模式对全部
        # 索引文献检测"未检索到相关片段"（no_relevant）
        if src_list:
            result["unrecalled_sources"] = self._zero_recall_sources(
                metadata_filter, result["retrieved"])
        elif use_balance:
            result["unrecalled_sources"] = self._auto_zero_recall(per_source)
        else:
            result["unrecalled_sources"] = []
        if save_log:
            self._append_log(result)
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_source_coverage(candidates: list, final: list, k: int) -> list:
        """
        均衡召回 × 重排 的组合保底。

        重排按分数取舍，可能让某篇文献的保底配额全军覆没（重排问题回到
        了"忽略短篇文献"）。这里把候选中未被 final 覆盖文献的最高分块
        换回：优先替换 final 中"块数>1 文献"的重排分最低条目，
        保证任何一篇有相关片段的文献至少有 1 块进入上下文。
        """
        from collections import Counter
        covered = Counter(h["chunk"].get("source") for h in final)
        final = list(final)
        for m in candidates:  # candidates 已按重排分降序
            src = m["chunk"].get("source")
            if covered.get(src, 0) > 0:
                continue  # 该文献已在 final 中
            if len(final) >= k:
                # 只牺牲"多块文献"中重排分最低的一块，避免清空其他文献
                victims = [h for h in final if covered.get(h["chunk"].get("source"), 0) > 1]
                if not victims:
                    break  # 全是单块来源，无法腾位
                victim = min(victims,
                             key=lambda h: h.get("rerank_score", h.get("score", 0)))
                covered[victim["chunk"].get("source")] -= 1
                final[final.index(victim)] = m
            else:
                final.append(m)
            covered[src] = covered.get(src, 0) + 1
        final.sort(key=lambda h: h.get("rerank_score", h.get("score", 0)), reverse=True)
        for r, h in enumerate(final, 1):
            h["rank"] = r
        return final

    # ------------------------------------------------------------------
    def _auto_zero_recall(self, per_source_counts: dict) -> list:
        """
        全库均衡模式下的零召回检测：索引中的某篇文献未检索到任何与
        问题相关的片段（与显式 source 过滤的 not_recalled 区分开，
        表示"问题与该文献内容相关性不足"而非"被其他文档挤出"）。
        """
        indexed = sorted({c.get("source") for c in self.retriever.chunks if c.get("source")})
        return [{"source": s, "reason": "no_relevant"}
                for s in indexed if per_source_counts.get(s, 0) == 0]

    # ------------------------------------------------------------------
    def _zero_recall_sources(self, metadata_filter, retrieved: list) -> list:
        """
        零召回检测：按 source 过滤查询时，找出未召回任何片段的文档。

        背景：单次检索的 top_k 名额有限，多文档联合查询时短文档容易被
        长文档的高分片段整体挤出召回。该检测把这一情况显式暴露出来，
        供终端告警与调用方处理。

        Returns:
            [{"source": 文件名, "reason": ...}, ...]，reason 取值:
            - "not_in_index": 该文件不在当前索引中（文件名拼错或未建索引）
            - "not_recalled": 已索引，但相关片段不足被其他文档挤出 top-k
            非多 source 查询（未带 source 过滤）返回空列表。
        """
        requested = (metadata_filter or {}).get("source")
        if not requested:
            return []
        if isinstance(requested, str):
            requested = [requested]
        # 兼容两种记录形状：ask() 规整化后的 {"source": ...} 与
        # 检索器原始命中 {"chunk": {"source": ...}}
        recalled = {
            r.get("source") or r.get("chunk", {}).get("source") for r in retrieved
        }
        indexed = {c.get("source") for c in self.retriever.chunks}
        out = []
        for s in requested:
            if s in recalled:
                continue
            out.append({
                "source": s,
                "reason": "not_in_index" if s not in indexed else "not_recalled",
            })
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _append_log(result: dict) -> None:
        """
        问答日志双写（任务④，均在 logs/ 文件夹）：
        1. logs/qa.jsonl —— 机器可读（时间戳/参数/全量字段，供评估与审计程序）；
        2. logs/qa.log   —— 人类可读（时间戳/参数/每条检索片段的来源与分数/
           回答全文），方便直接打开翻阅排查召回质量。
        """
        ensure_dirs()
        record = {"time": datetime.now().isoformat(timespec="seconds"), **result}
        with open(QA_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # ---- 人类可读日志 ----
        p = result.get("params", {})
        lines = ["=" * 70,
                 f"[{record['time']}] 问题: {result.get('question', '')}",
                 f"参数: {json.dumps(p, ensure_ascii=False)}",
                 "检索片段:"]
        for r in result.get("retrieved", []):
            rr = f" rerank={r['rerank_score']}" if r.get("rerank_score") is not None else ""
            lines.append(
                f"  ({r.get('rank')}) 检索分={r.get('score')}{rr} "
                f"{r.get('source', '?')} | {r.get('section', '?')} "
                f"p{r.get('page', '?')} [{r.get('chunk_id', '')}]")
        unrecalled = result.get("unrecalled_sources") or []
        if unrecalled:
            lines.append("未召回: " + ", ".join(
                f"{u['source']}({u['reason']})" for u in unrecalled))
        lines += ["回答:", result.get("answer", ""), ""]
        with open(QA_TEXT_LOG, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))


# ==========================================================================
# 交互式命令行
# ==========================================================================
def _print_result(res: dict) -> None:
    """在终端友好地展示答案与引用溯源。"""
    print("\n" + "=" * 70)
    print("【答案】")
    print(res["answer"])
    if res["insufficient"]:
        print(">> 提示：模型判定文献证据不足，请尝试换一种问法或补充相关论文。")
    if res["invalid_markers"]:
        print(f">> 警告：答案引用了不存在的编号 {res['invalid_markers']}（潜在幻觉信号）。")
    if res["citations"]:
        print("\n【引用来源】")
        for c in res["citations"]:
            print(f"  [{c['marker']}] {c.get('source') or '未知文件'} | "
                  f"《{c['paper_title']}》 "
                  f"{section_zh(c['section'])}部分 第{c['page']}页 "
                  f"(chunk: {c['chunk_id']})")
    print("=" * 70)


def main() -> None:
    setup_console()
    ensure_dirs()
    import sys
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        _print_result(RAGPipeline().ask(question))
        return
    # 无参数 -> 进入交互问答循环
    print("学术文献 RAG 研读助手（输入问题开始，直接回车退出）")
    rag = RAGPipeline()
    while True:
        try:
            question = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not question:
            break
        try:
            _print_result(rag.ask(question))
        except Exception as e:
            print(f"[错误] {e}")


if __name__ == "__main__":
    main()
