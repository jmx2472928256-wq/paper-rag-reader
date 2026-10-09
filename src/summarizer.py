# -*- coding: utf-8 -*-
"""
结构化笔记生成模块 —— 文献阅读工具的核心。

设计目标（预生成优先）
--------------------
先为每篇 PDF 预生成固定模板的结构化摘要（Markdown），日常阅读以笔记为主；
RAG 问答（ask）仅用于深挖原文细节。这样规避了每次问答都依赖实时向量
召回的不稳定性，也避免多文档联合检索时短文档被长文档挤出召回。

功能
----
1. generate_summary(source, rag): 针对单篇 PDF，按模板字段（作者/研究
   问题/方法/数据/结论/局限）分组定向检索证据，调用 GLM-5.3 生成固定
   模板摘要，保存到 literature_lib/summary/<主文件名>.md；
2. compare_summaries(sources, rag): 读取（缺失时先补生成）各篇摘要，
   调用 LLM 输出横向对比 Markdown，保存到 literature_lib/compare/；
3. zero_recall_warn_lines / build_qa_markdown: 多文档零召回的醒目警告
   行与问答结果 Markdown 组装（终端输出与批处理保存共用）。
"""

import hashlib
import re
from datetime import datetime
from pathlib import Path

try:
    from .config import (
        ANSWERS_DIR, COMPARE_DIR, LLM_MODEL, SUMMARY_DIR, chat, clamp_top_k,
        ensure_dirs, setup_console,
    )
except ImportError:  # 以脚本方式直接运行
    from config import (
        ANSWERS_DIR, COMPARE_DIR, LLM_MODEL, SUMMARY_DIR, chat, clamp_top_k,
        ensure_dirs, setup_console,
    )

# ==========================================================================
# 摘要固定模板（字段名与顺序不得改动）
# ==========================================================================
SUMMARY_TEMPLATE = """【{source} 文献摘要】

标题：
作者：
研究问题：
研究方法：
数据 / 样本：
主要结论：
研究局限："""

# 摘要必含的 7 个字段（用于生成后校验）
_SUMMARY_FIELDS = ("标题", "作者", "研究问题", "研究方法", "数据 / 样本", "主要结论", "研究局限")

# 模板各字段 -> 定向证据检索查询词（逐字段召回比单查询覆盖更全面，
# 每字段独立一次查询，共 6 次 embedding 调用 + 1 次 LLM 调用）
FIELD_QUERIES = [
    ("作者", "论文作者 学位申请人 指导教师 作者单位 姓名", 3),
    ("研究问题", "研究问题 研究目标 研究背景 研究动机", 4),
    ("研究方法", "研究方法 预测模型 技术路线 算法框架", 4),
    ("数据 / 样本", "数据 样本 数据集 实验数据 变量 数据来源", 4),
    ("主要结论", "主要结论 实验结果 研究发现 效果", 4),
    ("研究局限", "研究局限 不足 限制 未来研究 研究展望", 3),
]

# 每条证据进入提示词的最大字符数（控制成本，摘要不需要整块原文）
_EVIDENCE_SNIPPET_CHARS = 350

_SUMMARY_SYSTEM_PROMPT = """你是严谨的学术文献笔记助手。请仅依据【证据】部分填写文献摘要模板，规则：
1. 每个字段只使用证据中明确出现的信息，禁止编造作者、数字、结论；
2. 某字段证据不足时，该字段填写"未在文献中检索到相关信息"；
3. 输出第一行必须是【{source} 文献摘要】，随后按模板字段顺序输出，字段名保持一致；
4. 内容具体、简洁，可用分点或短句。"""

# 多文档零召回时追加到输出末尾的对比建议
COMPARE_TIP = ("💡 多篇文献对比建议分开生成各篇摘要（--summary），"
               "使用 compare 子命令完成对比，避免召回缺失。")


# ==========================================================================
# 摘要生成
# ==========================================================================
def _gather_evidence(source: str, rag):
    """
    按模板字段分组定向检索证据。

    Args:
        source: 原始 PDF 文件名。
        rag: RAGPipeline 实例（复用其检索器与 references 排除白名单）。

    Returns:
        (groups, seen): groups 为 [(字段名, [chunk, ...]), ...]；
        seen 为全部用到的 chunk_id 集合（全局去重，避免同一块重复占上下文）。
    """
    groups, seen = [], set()
    for label, query, top_k in FIELD_QUERIES:
        hits = rag.retriever.search(
            query, top_k=top_k, mode="hybrid",
            allowed_sections=rag.allowed_sections,
            metadata_filter={"source": source},
        )
        picked = []
        for h in hits:
            if h["chunk_id"] in seen:
                continue
            seen.add(h["chunk_id"])
            picked.append(h["chunk"])
        groups.append((label, picked))
    return groups, seen


def _format_evidence(groups, paper_title: str) -> str:
    """把分组证据格式化为提示词文本（每条截断到固定长度控制成本）。"""
    lines = [f"索引元数据标题: {paper_title}", ""]
    for label, chunks in groups:
        lines.append(f"[证据-{label}]")
        if not chunks:
            lines.append("(未检索到相关片段)")
        for c in chunks:
            text = c.get("text", "")[:_EVIDENCE_SNIPPET_CHARS]
            lines.append(f"- (第{c.get('page', '?')}页) {text}")
        lines.append("")
    return "\n".join(lines)


def generate_summary(source: str, rag, save: bool = True) -> dict:
    """
    为单篇 PDF 生成固定模板的结构化文献摘要。

    Args:
        source: 原始 PDF 文件名（须已在索引中）。
        rag: RAGPipeline 实例。
        save: 是否保存 Markdown 到 literature_lib/summary/<主文件名>.md。

    Returns:
        {"source", "markdown", "answer", "chunk_ids", "path"}
    """
    ensure_dirs()
    # 校验该文件确在索引中，报错时给出可选文件名（防止拼写错误静默失败）
    indexed = sorted(x for x in {c.get("source") for c in rag.retriever.chunks} if x)
    if source not in indexed:
        raise ValueError(f"{source} 不在当前索引中。可用文件名: {indexed}")

    groups, seen = _gather_evidence(source, rag)
    paper_title = next((c.get("paper_title", "") for _, chs in groups for c in chs), source)
    evidence = _format_evidence(groups, paper_title)

    messages = [
        {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT.format(source=source)},
        {"role": "user", "content": (
            f"【证据】\n{evidence}\n\n"
            f"请严格按以下模板输出摘要：\n{SUMMARY_TEMPLATE.format(source=source)}"
        )},
    ]
    answer = chat(messages, temperature=0.1)

    # 轻量校验：模板字段是否齐全（缺失只告警不重试，多半是原文无相关信息）
    missing = [f for f in _SUMMARY_FIELDS if f"{f}：" not in answer and f"{f}:" not in answer]
    if missing:
        print(f"[summarizer] 警告: 摘要缺少字段 {missing}（可能是原文无相关信息或输出异常）")

    now = datetime.now().isoformat(timespec="seconds")
    markdown = (
        answer.rstrip()
        + f"\n\n---\n> 证据分块 {len(seen)} 个: {', '.join(sorted(seen))}  \n"
          f"> 生成时间: {now} | 模型: {LLM_MODEL}"
    )
    path = None
    if save:
        SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
        path = SUMMARY_DIR / f"{Path(source).stem}.md"
        path.write_text(markdown, encoding="utf-8")
    return {"source": source, "markdown": markdown, "answer": answer,
            "chunk_ids": sorted(seen), "path": path}


# ==========================================================================
# 多篇横向对比
# ==========================================================================
def compare_summaries(sources: list, rag, refresh: bool = False, save: bool = True) -> dict:
    """
    横向对比多篇文献：读取已保存摘要（缺失时先自动生成），调用 LLM
    输出对比 Markdown 并保存到 literature_lib/compare/。

    Returns:
        {"markdown", "path", "generated"}，generated 为本次新生成摘要的文件名。
    """
    ensure_dirs()
    if len(sources) < 2:
        raise ValueError("compare 至少需要两篇文献。")
    parts, generated = [], []
    for s in sources:
        path = SUMMARY_DIR / f"{Path(s).stem}.md"
        if path.exists() and not refresh:
            content = path.read_text(encoding="utf-8")
        else:
            res = generate_summary(s, rag)
            content = res["markdown"]
            generated.append(s)
        parts.append(f"## {s}\n\n{content}")

    prompt = (
        f"以下是{len(sources)}篇文献的结构化摘要，请输出横向对比 Markdown，包含：\n"
        "1) 对比表（维度：研究问题/研究方法/数据或样本/主要结论/研究局限）；\n"
        "2) 共性要点；3) 差异要点；4) 一句话总评。\n"
        "仅基于摘要内容对比，不得补充外部信息。\n\n" + "\n\n".join(parts)
    )
    answer = chat([{"role": "user", "content": prompt}], temperature=0.1)

    now = datetime.now().isoformat(timespec="seconds")
    md = (f"# 文献对比: {' vs '.join(sources)}\n\n{answer}"
          f"\n\n---\n> 基于各篇结构化摘要生成 | 生成时间: {now}")
    path = None
    if save:
        COMPARE_DIR.mkdir(parents=True, exist_ok=True)
        path = COMPARE_DIR / f"{'_vs_'.join(Path(s).stem for s in sources)}.md"
        path.write_text(md, encoding="utf-8")
    return {"markdown": md, "path": path, "generated": generated}


# ==========================================================================
# 问题驱动的多篇对比（「对比」标签页后端：逐篇限定检索 -> 单篇观点 -> 汇总）
# ==========================================================================
# 单篇观点总结的约束：仅依据本篇证据、不足时如实说明、带编号引用
_VIEWPOINT_SYSTEM_PROMPT = """你是严谨的学术文献研读助手。请仅依据【证据】回答【问题】中针对本篇文献的部分：
1. 只使用证据中的信息，禁止引入外部知识或编造内容；
2. 证据不足时必须明确说明"本篇文献中未检索到与该问题直接相关的内容"，不得强行作答；
3. 关键论断用 [编号] 标注证据序号；数字与原文完全一致；
4. 200字以内，分点陈述，聚焦本篇文献自身的观点/方法/结论。"""

# 汇总对比的约束：只依据各篇观点总结横向对比，不引入外部信息
_AGGREGATE_SYSTEM_PROMPT = """你是学术文献对比分析助手。你只能依据用户提供的各篇文献观点总结进行横向对比：
1. 禁止引入外部知识，禁止补充各篇观点总结之外的内容；
2. 输出 Markdown，依次包含：1) 对比表（维度依据问题自行确定，含"与问题的关联"）2) 共性要点 3) 差异要点 4) 综合结论；
3. 某篇观点标注"未检索到相关内容"时，在对比表中如实标注"证据不足"，不得强行对比。"""


# ==========================================================================
# 自动对比模式（问题为空时）：五个固定维度，检索词复用 FIELD_QUERIES
# ==========================================================================
AUTO_QUESTION_TEXT = "文献内容全面对比（研究背景、研究方法、数据集、核心结论、局限性）"
# (维度名, 检索查询词)：查询词直接取自 FIELD_QUERIES 的对应字段（复用既有调优）
AUTO_DIMENSIONS = [
    ("研究背景", next(q for label, q, _ in FIELD_QUERIES if label == "研究问题")),
    ("研究方法", next(q for label, q, _ in FIELD_QUERIES if label == "研究方法")),
    ("数据集", next(q for label, q, _ in FIELD_QUERIES if label == "数据 / 样本")),
    ("核心结论", next(q for label, q, _ in FIELD_QUERIES if label == "主要结论")),
    ("局限性", next(q for label, q, _ in FIELD_QUERIES if label == "研究局限")),
]
# 自动模式单篇观点总结的维度指令
_AUTO_VIEWPOINT_TASK = (
    "请从以下五个维度逐项总结本篇文献：研究背景、研究方法、数据集、核心结论、局限性。\n"
    "每个维度单独一行小标题；该维度证据不足时写「本维度证据不足」，禁止编造。"
)


def compare_viewpoints(question: str, sources: list, rag,
                       top_k_per_paper: int = None,
                       use_saved_summaries: bool = False,
                       save: bool = True) -> dict:
    """
    多篇文献横向对比（「对比」标签页后端，双模式）。

    - 问题驱动模式：question 非空，围绕用户问题逐篇检索并对比（原有逻辑）；
    - 自动对比模式：question 为空且勾选 >=2 篇，自动按五个固定维度
      （研究背景/研究方法/数据集/核心结论/局限性）逐篇定向检索、
      生成单篇五维度总结，再做整体横向对比。

    流程（全程复用现有 FAISS 混合检索与 GLM-5.3 调用，不新增依赖）：
    1. 对每篇被勾选的文献做限定来源检索（rag.retrieve + metadata_filter）；
    2. 每篇调用一次 GLM-5.3 生成"单篇观点总结"（带 [编号] 引用，
       证据不足时如实标注「证据不足」，禁止编造）；
    3. 汇总各篇观点，再一次调用生成横向对比 Markdown，保存到
       literature_lib/compare/。

    Args:
        question: 对比问题；空/None 时进入自动对比模式。
        sources: 参与对比的 PDF 文件名列表（至少 2 篇，且须在索引中）。
        rag: RAGPipeline 实例。
        top_k_per_paper: 每篇/每维度检索的证据块数（默认 4，经 clamp_top_k 钳制）。
        use_saved_summaries: 是否把各篇已保存的结构化摘要一并纳入证据。
        save: 是否保存 Markdown。

    Returns:
        {"question", "mode", "sources", "markdown", "path", "per_paper", "failed"}
    """
    ensure_dirs()

    # ---- 边界校验：文献数不足 / 文献不在索引中（问题为空是合法的自动模式）----
    question = (question or "").strip()
    auto_mode = not question
    if auto_mode:
        question = AUTO_QUESTION_TEXT
    sources = [str(s).strip() for s in (sources or []) if str(s).strip()]
    if len(sources) < 2:
        raise ValueError("对比至少需要两篇文献")
    indexed = sorted({c.get("source") for c in rag.retriever.chunks if c.get("source")})
    unknown = [s for s in sources if s not in indexed]
    if unknown:
        raise ValueError(f"以下文献不在索引中: {unknown}（可用: {indexed}）")

    k = clamp_top_k(top_k_per_paper, 4)  # 每篇/每维度证据块数（复用边界保护）

    # ---- 第一步：逐篇限定来源检索 + 单篇观点总结 ----
    per_paper, failed = [], []
    for s in sources:
        mode_desc = "自动五维度" if auto_mode else "问题定向"
        print(f"[summarizer] 观点对比({mode_desc}): 检索并总结 {s} ...")
        try:
            paper_title = ""
            if auto_mode:
                # 自动模式：按五个维度分别定向检索（查询词复用 FIELD_QUERIES）
                groups = []
                for dim, dim_query in AUTO_DIMENSIONS:
                    hits, _info = rag.retrieve(dim_query, top_k=k,
                                               metadata_filter={"source": s},
                                               save_log=True)
                    chunks = [h.get("chunk", {}) for h in hits]
                    groups.append((dim, chunks))
                    if not paper_title:
                        paper_title = next((c.get("paper_title", "") for c in chunks
                                            if c.get("paper_title")), "")
                evidence = _format_evidence(groups, paper_title or s)
                user_task = (f"【分析任务】\n{_AUTO_VIEWPOINT_TASK}\n\n"
                             f"【证据】（来自 {s}，按维度分组）\n{evidence}")
            else:
                # 问题驱动模式（原有逻辑不变）
                hits, _info = rag.retrieve(question, top_k=k,
                                           metadata_filter={"source": s},
                                           save_log=True)
                chunks = [h.get("chunk", {}) for h in hits]
                paper_title = next((c.get("paper_title", "") for c in chunks
                                    if c.get("paper_title")), s)
                evidence = _format_evidence([("证据", chunks)], paper_title)
                user_task = (f"【问题】\n{question}\n\n"
                             f"【证据】（来自 {s}，共{len(chunks)}块）\n{evidence}")
            if use_saved_summaries:
                saved_path = SUMMARY_DIR / f"{Path(s).stem}.md"
                if saved_path.exists():
                    excerpt = saved_path.read_text(encoding="utf-8")[:800]
                    user_task += f"\n\n[附-已保存摘要摘录]\n{excerpt}"
            view = chat(
                [{"role": "system", "content": _VIEWPOINT_SYSTEM_PROMPT},
                 {"role": "user", "content": user_task}],
                temperature=0.1)
            per_paper.append({"source": s, "title": paper_title or "",
                              "mode": "auto" if auto_mode else "question",
                              "viewpoint": view})
        except Exception as e:  # 单篇失败不中断整体对比
            failed.append({"source": s, "error": str(e)})
            per_paper.append({"source": s, "title": "",
                              "viewpoint": f"（该篇观点总结生成失败: {e}）"})

    # ---- 第二步：汇总各篇观点，生成横向对比 ----
    parts = []
    for pp in per_paper:
        t = f"《{pp['title']}》" if pp.get("title") else ""
        parts.append(f"### {pp['source']} {t}\n\n{pp['viewpoint']}")
    if auto_mode:
        agg_user = ("【对比任务】\n自动模式：请围绕五个固定维度进行横向对比，"
                    "对比表的行必须是：研究背景、研究方法、数据集、核心结论、局限性。\n\n"
                    "【各篇文献观点总结】\n\n" + "\n\n".join(parts))
    else:
        agg_user = (f"【对比问题】\n{question}\n\n"
                    "【各篇文献观点总结】\n\n" + "\n\n".join(parts))
    aggregate = chat(
        [{"role": "system", "content": _AGGREGATE_SYSTEM_PROMPT},
         {"role": "user", "content": agg_user}],
        temperature=0.1)

    now = datetime.now().isoformat(timespec="seconds")
    stems = "_vs_".join(Path(s).stem for s in sources)
    if auto_mode:
        # 自动模式问题固定，用时间戳避免互相覆盖；标题区分模式
        fname = f"自动对比_{stems}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        title = f"# 自动对比: {' vs '.join(sources)}"
    else:
        qhash = hashlib.md5(question.encode("utf-8")).hexdigest()[:6]
        fname = f"观点对比_{stems}_{qhash}.md"
        title = f"# 观点对比: {' vs '.join(sources)}"
    md = (f"{title}\n\n"
          f"**对比任务**: {question}\n\n{aggregate}"
          + "\n\n---\n\n## 附录：各篇单篇观点总结\n\n" + "\n\n".join(parts)
          + f"\n\n---\n> 模式: {'自动(五维度)' if auto_mode else '问题驱动'} | "
            f"每篇/每维度检索 {k} 块证据 | 模型: {LLM_MODEL} | 生成时间: {now}")
    if failed:
        md += f"\n> ⚠️ 以下文献的观点总结生成失败，未纳入对比: " \
              f"{', '.join(f['source'] for f in failed)}"

    path = None
    if save:
        COMPARE_DIR.mkdir(parents=True, exist_ok=True)
        path = COMPARE_DIR / fname
        path.write_text(md, encoding="utf-8")
        print(f"[summarizer] 对比报告已保存 -> {path}"
              + (f"（{len(failed)}篇生成失败）" if failed else ""))
    return {"question": question, "mode": "auto" if auto_mode else "question",
            "sources": sources, "markdown": md, "path": path,
            "per_paper": per_paper, "failed": failed}


# ==========================================================================
# 零召回警告与问答 Markdown 组装（终端输出与批处理保存共用）
# ==========================================================================
def zero_recall_warn_lines(res: dict, multi_source: bool = False) -> list:
    """
    多 source 查询出现零召回文档时，生成输出内容头部的醒目警告行。

    Returns:
        ["⚠️警告：xxx.pdf 本次检索无召回片段，回答仅基于其余文档证据", ...]
        非 multi_source 或无零召回时返回空列表。
    """
    if not multi_source:
        return []
    lines = []
    for item in res.get("unrecalled_sources") or []:
        if item["reason"] == "not_in_index":
            lines.append(f"⚠️警告：{item['source']} 不在当前索引中，回答仅基于其余文档证据")
        else:
            lines.append(f"⚠️警告：{item['source']} 本次检索无召回片段，回答仅基于其余文档证据")
    return lines


def build_qa_markdown(res: dict, warn_lines=None, tail_note=None,
                      metadata_filter=None) -> str:
    """
    将 ask 问答结果组装为 Markdown（批处理保存用，与终端展示内容一致）。

    Args:
        res: rag.ask() 返回的结果 dict。
        warn_lines: 头部零召回警告行（见 zero_recall_warn_lines）。
        tail_note: 末尾追加的提示（如 COMPARE_TIP）。
        metadata_filter: 本次检索的过滤条件（用于展示检索范围）。
    """
    lines = []
    for w in (warn_lines or []):
        lines.append(w)
    if warn_lines:
        lines.append("")

    lines += ["# 问答记录", ""]
    srcs = (metadata_filter or {}).get("source")
    if srcs:
        srcs = [srcs] if isinstance(srcs, str) else list(srcs)
        lines += [f"**检索范围**: {', '.join(srcs)}", ""]
    lines += [f"**问题**: {res.get('question', '')}", ""]
    counts = res.get("per_source_counts") or {}
    if counts:
        lines.append("**文献覆盖**: " + "  ".join(f"{s}×{n}块" for s, n in counts.items()))
        lines.append("")
    params = res.get("params") or {}
    if params:
        kv = ", ".join(f"{k}={v}" for k, v in params.items() if v is not None)
        lines += [f"**参数**: {kv}", ""]
    lines += ["## 回答", "", res.get("answer", ""), ""]

    cites = res.get("citations") or []
    if cites:
        lines += ["## 引用来源", ""]
        for c in cites:
            lines.append(f"- [{c['marker']}] {c.get('source', '?')} | "
                         f"《{c.get('paper_title', '')}》 "
                         f"{c.get('section', '')} 第{c.get('page', '?')}页")
        lines.append("")

    unrecalled = res.get("unrecalled_sources") or []
    if unrecalled:
        lines += ["## 零召回文档", ""]
        for item in unrecalled:
            reason = "不在索引中" if item["reason"] == "not_in_index" else "未召回任何片段"
            lines.append(f"- {item['source']}（{reason}）")
        lines.append("")

    if tail_note:
        lines += ["---", "", tail_note]
    return "\n".join(lines)


if __name__ == "__main__":
    setup_console()
    print("本模块为库模块，请通过 python -m src.main ask --summary / compare 调用。")
