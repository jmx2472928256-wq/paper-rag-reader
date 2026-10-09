# -*- coding: utf-8 -*-
"""
评估模块：召回率 / 问答准确率 / 幻觉率 / 消融实验。

评估集格式（JSON 文件，默认 experiments/eval_set.json）
-------------------------------------------------------
{
  "queries": [
    {
      "question": "Attention机制的核心思想是什么？",
      "relevant_chunk_ids": ["a1b2-sen-0003"],   # 可选：人工标注的相关块
      "gold_section": "method",                   # 可选：粗标注相关章节
      "paper_id": "a1b2c3d4e5f6",                 # 可选：限定论文范围
      "reference_answer": "……",                   # 准确率评估的标准答案
      "keywords": ["自注意力", "QKV"]             # 准确率评估的关键词
    }, ...
  ]
}
字段按评估类型按需提供：召回率需要 relevant_chunk_ids 或 gold_section；
准确率/幻觉率需要 question（其余字段可选，取决于评估模式）。

评估口径
--------
1. 召回率 Recall@k：检索 top_k 中命中"金标块集合"的比例，多查询取均值；
2. 问答准确率：
   - keyword 模式：标准答案关键词的命中率 >= 阈值判为正确（无需调用 LLM）；
   - llm 模式：调用 GLM-5.3 以标准答案为参照做 0/1 判卷；
3. 幻觉率：将答案拆成论断句，逐条判断是否被检索证据支持：
   - llm 模式：GLM-5.3 逐条事实核查；
   - citation 模式：无 API 时的代理指标——含有引用标记的论断视为
     有证据支撑，未带标记的事实性论断视为无支撑（保守估计）；
4. 消融实验：固定评估集，切换 {bm25 / dense / hybrid(weighted) /
   hybrid(rrf)} 等检索配置，对比召回率，结果落盘 experiments/。

用法
----
    python -m src.evaluator --mode recall --eval-file experiments/eval_set.json
    python -m src.evaluator --mode ablation
    python -m src.evaluator --mode qa --judge llm
    python -m src.evaluator --mode all --with-qa
"""

import json
import re
from pathlib import Path

try:
    from . import rag_pipeline, retrieval
    from .config import EXPERIMENTS_DIR, INDEX_DIR, chat, ensure_dirs, setup_console
except ImportError:  # 以脚本方式直接运行
    import rag_pipeline
    import retrieval
    from config import EXPERIMENTS_DIR, INDEX_DIR, chat, ensure_dirs, setup_console

DEFAULT_EVAL_FILE = EXPERIMENTS_DIR / "eval_set.json"
ABLATION_OUT_FILE = EXPERIMENTS_DIR / "ablation_results.json"

# 消融实验默认配置：覆盖检索模式与融合策略两个维度
DEFAULT_ABLATION_CONFIGS = [
    {"name": "bm25_only",       "mode": "bm25"},
    {"name": "dense_only",      "mode": "dense"},
    {"name": "hybrid_weighted", "mode": "hybrid", "fusion": "weighted", "alpha": 0.5},
    {"name": "hybrid_rrf",      "mode": "hybrid", "fusion": "rrf"},
]


# ==========================================================================
# 评估集加载
# ==========================================================================
def load_eval_file(eval_file) -> list:
    """读取评估集 JSON，返回 queries 列表；文件不存在时生成模板并中止。"""
    eval_file = Path(eval_file)
    if not eval_file.exists():
        template_path = make_eval_template(eval_file)
        raise FileNotFoundError(
            f"评估集不存在: {eval_file}\n已生成模板 {template_path}，"
            "请按模板填写评估问题后重试。")
    data = json.loads(eval_file.read_text(encoding="utf-8"))
    queries = data.get("queries", []) if isinstance(data, dict) else data
    if not queries:
        raise ValueError(f"评估集为空: {eval_file}")
    return queries


def make_eval_template(path=None) -> Path:
    """生成带字段说明的评估集模板（首次运行评估时自动创建）。"""
    path = Path(path or DEFAULT_EVAL_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    template = {
        "_说明": {
            "relevant_chunk_ids": "召回率金标：人工标注的相关块ID列表（可从检索日志中挑选）",
            "gold_section": "召回率粗金标：相关章节规范名(abstract/introduction/method/experiment/results/conclusion)",
            "paper_id": "可选：限定金标所在论文（不填则匹配全部论文同章节）",
            "reference_answer": "准确率评估（llm模式）所需的标准答案",
            "keywords": "准确率评估（keyword模式）所需的关键词列表",
        },
        "queries": [
            {
                "question": "示例：本文提出的方法在哪些数据集上进行了验证？",
                "relevant_chunk_ids": [],
                "gold_section": "experiment",
                "paper_id": "",
                "reference_answer": "示例标准答案：在 XXX 数据集上验证。",
                "keywords": ["数据集"],
            }
        ],
    }
    path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[evaluator] 已生成评估集模板 -> {path}")
    return path


# ==========================================================================
# 1. 召回率
# ==========================================================================
def _resolve_gold_ids(chunks: list, query: dict):
    """
    解析一条查询的金标块集合。

    优先级: relevant_chunk_ids > (gold_section + paper_id) > paper_id 全文。
    无法解析返回 None（该条跳过并告警）。

    注意：gold_section 粗金标会展开为该章节全部粒度的块（含句子级），
    Recall@k 口径偏严格；更公平的做法是标注 relevant_chunk_ids。
    """
    ids = set(query.get("relevant_chunk_ids") or [])
    if ids:
        return ids
    paper_id = (query.get("paper_id") or "").strip()
    gold_section = (query.get("gold_section") or "").strip()
    if gold_section:
        return {c["chunk_id"] for c in chunks
                if c.get("section") == gold_section
                and (not paper_id or c.get("paper_id") == paper_id)}
    if paper_id:
        return {c["chunk_id"] for c in chunks if c.get("paper_id") == paper_id}
    return None


def eval_recall(retriever, queries: list, top_k: int = 5, mode: str = "hybrid",
                fusion: str = "weighted", alpha: float = 0.5) -> dict:
    """
    计算 Recall@k（每条查询: |检索命中 ∩ 金标| / |金标|，均值聚合）。
    """
    details, skipped = [], 0
    for q in queries:
        gold = _resolve_gold_ids(retriever.chunks, q)
        if not gold:
            skipped += 1
            details.append({"question": q["question"], "skipped": True,
                            "reason": "缺少金标(relevant_chunk_ids/gold_section/paper_id)"})
            continue
        hits = retriever.search(q["question"], top_k=top_k, mode=mode,
                                fusion=fusion, alpha=alpha)
        got = {h["chunk_id"] for h in hits}
        hit_gold = got & gold
        details.append({
            "question": q["question"],
            "recall": round(len(hit_gold) / len(gold), 4),
            "num_gold": len(gold),
            "num_retrieved": len(got),
            "hit_gold": sorted(hit_gold),
        })
    scored = [d["recall"] for d in details if not d.get("skipped")]
    return {
        "metric": f"recall@{top_k}",
        "config": {"mode": mode, "fusion": fusion, "alpha": alpha},
        "mean_recall": round(sum(scored) / len(scored), 4) if scored else None,
        "num_queries": len(queries), "num_skipped": skipped,
        "details": details,
    }


# ==========================================================================
# 2. 问答准确率
# ==========================================================================
def _judge_answer_llm(generated: str, reference: str) -> bool:
    """GLM-5.3 判卷：以标准答案为参照，模型回答含关键信息且无事实错误记 1 分。"""
    prompt = (
        "你是问答阅卷员。下面给出标准答案与模型回答，请判断模型回答是否正确：\n"
        "判定标准：模型回答覆盖了标准答案的关键信息，且没有与标准答案矛盾的内容。\n"
        f"\n【标准答案】\n{reference}\n\n【模型回答】\n{generated}\n\n"
        "只输出一个数字：正确输出 1，错误输出 0。"
    )
    ans = chat([{"role": "user", "content": prompt}], temperature=0.0)
    return ans.strip().startswith("1")


def eval_qa(queries: list, rag=None, judge: str = "keyword",
            keyword_threshold: float = 0.5, top_k: int = 6,
            mode: str = "hybrid") -> dict:
    """
    问答准确率评估（需先构建索引；llm 模式还需配置 API Key）。

    Args:
        rag: RAGPipeline 实例；为 None 时自动构建。
        judge: "keyword"（关键词命中率，离线）或 "llm"（GLM-5.3 判卷）。
        keyword_threshold: keyword 模式的正确判定阈值。
    """
    rag = rag or rag_pipeline.RAGPipeline()
    details, num_correct = [], 0
    for q in queries:
        # 评估题若标注了所属论文（paper_id），按单篇检索作答，与
        # "单独提问其中一篇只答该篇"的语义保持一致；
        # rerank=False 固定关闭重排，保证评估结果与历史口径可比
        mf = {"source": q["paper_id"]} if q.get("paper_id") else None
        result = rag.ask(q["question"], top_k=top_k, mode=mode, save_log=False,
                         metadata_filter=mf, rerank=False)
        generated = result["answer"]
        record = {"question": q["question"], "generated": generated,
                  "insufficient": result["insufficient"]}
        if judge == "llm":
            reference = q.get("reference_answer", "")
            if not reference:
                record.update({"skipped": True, "reason": "缺少 reference_answer"})
            else:
                correct = _judge_answer_llm(generated, reference)
                num_correct += int(correct)
                record["correct"] = correct
        else:  # keyword 模式
            keywords = q.get("keywords") or []
            if not keywords:
                record.update({"skipped": True, "reason": "缺少 keywords"})
            else:
                hit = [kw for kw in keywords if kw.lower() in generated.lower()]
                ratio = len(hit) / len(keywords)
                correct = ratio >= keyword_threshold
                num_correct += int(correct)
                record.update({"correct": correct, "keyword_hit_ratio": round(ratio, 3),
                               "hit_keywords": hit})
        details.append(record)
    scored = [d for d in details if not d.get("skipped")]
    return {
        "metric": "qa_accuracy",
        "config": {"judge": judge, "top_k": top_k, "mode": mode,
                   "keyword_threshold": keyword_threshold},
        "accuracy": round(num_correct / len(scored), 4) if scored else None,
        "num_queries": len(queries), "num_skipped": len(queries) - len(scored),
        "details": details,
    }


# ==========================================================================
# 3. 幻觉率
# ==========================================================================
_CLAIM_SPLIT_RE = re.compile(r"[。！？!?\n]+")


def _split_claims(answer: str, min_len: int = 8, max_claims: int = 20) -> list:
    """
    将答案拆为论断句（幻觉率评估的基本单元）。
    过滤过短碎片与列表引导行，超出 max_claims 截断（控制判卷成本）。
    """
    claims = [c.strip(" -•*0123456789.、") for c in _CLAIM_SPLIT_RE.split(answer)]
    claims = [c for c in claims if len(c) >= min_len and not c.startswith(("【", "[", "("))]
    return claims[:max_claims]


def _judge_claims_llm(claims: list, context_texts: list) -> list:
    """
    GLM-5.3 事实核查：判断每条论断是否被检索证据直接支持。
    单次调用批量判卷，返回与 claims 对齐的布尔列表。
    """
    context = "\n\n".join(f"[证据{i + 1}]\n{t[:400]}" for i, t in enumerate(context_texts))
    claim_list = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    prompt = (
        "你是事实核查员。依据下面的【证据】判断每条论断是否被证据直接支持"
        "（证据中能找到明确依据才算支持；与证据矛盾或证据未提及均算不支持）。\n\n"
        f"【证据】\n{context}\n\n【论断】\n{claim_list}\n\n"
        '只输出 JSON 数组，格式: [{"id": 1, "supported": true}, {"id": 2, "supported": false}]'
    )
    ans = chat([{"role": "user", "content": prompt}], temperature=0.0)
    # 容错解析：优先整体 JSON，失败则用正则逐条抓取
    supported = [False] * len(claims)
    matches = re.findall(r'"id"\s*:\s*(\d+)[^{}]*?"supported"\s*:\s*(true|false)', ans, re.IGNORECASE)
    for idx_str, flag in matches:
        i = int(idx_str) - 1
        if 0 <= i < len(claims):
            supported[i] = flag.lower() == "true"
    return supported


def eval_hallucination(queries: list, rag=None, judge: str = "llm",
                       top_k: int = 6, mode: str = "hybrid") -> dict:
    """
    幻觉率评估：幻觉率 = 不被证据支持的论断数 / 论断总数。

    - judge="llm": GLM-5.3 逐条事实核查（推荐，需 API Key）；
    - judge="citation": 代理指标——论断含 [n] 引用标记视为有支撑，
      否则视为无支撑（离线可算，仅作下界参考）。
    """
    rag = rag or rag_pipeline.RAGPipeline()
    details, total_claims, total_unsupported = [], 0, 0
    for q in queries:
        mf = {"source": q["paper_id"]} if q.get("paper_id") else None
        result = rag.ask(q["question"], top_k=top_k, mode=mode, save_log=False,
                         metadata_filter=mf, rerank=False)
        answer = result["answer"]
        claims = _split_claims(answer)
        if not claims:
            details.append({"question": q["question"], "num_claims": 0,
                            "unsupported": 0, "note": "无法拆分论断（可能是拒答）"})
            continue
        if judge == "llm":
            evidence = [r["text"] for r in result["retrieved"]]
            supported = _judge_claims_llm(claims, evidence)
        else:  # citation 代理：带引用标记的论断记为有支撑
            supported = [bool(re.search(r"\[\d{1,2}\]", c)) for c in claims]
        unsupported = sum(1 for s in supported if not s)
        total_claims += len(claims)
        total_unsupported += unsupported
        details.append({
            "question": q["question"], "num_claims": len(claims),
            "unsupported": unsupported,
            "unsupported_claims": [c for c, s in zip(claims, supported) if not s],
            "answer": answer,
        })
    rate = round(total_unsupported / total_claims, 4) if total_claims else None
    return {
        "metric": "hallucination_rate",
        "config": {"judge": judge, "top_k": top_k, "mode": mode},
        "hallucination_rate": rate,
        "total_claims": total_claims, "total_unsupported": total_unsupported,
        "num_queries": len(queries), "details": details,
    }


# ==========================================================================
# 4. 消融实验
# ==========================================================================
def run_ablation(queries: list, retriever=None, top_k: int = 5,
                 configs: list = None, out_file=ABLATION_OUT_FILE) -> dict:
    """
    检索消融实验：同一评估集上对比不同检索配置的 Recall@k。
    索引只加载一次，多配置复用（BM25/向量均无需重建）。

    Returns:
        {"results": {配置名: 评估结果}, "best_config": 配置名}
        并写入 experiments/ablation_results.json。
    """
    retriever = retriever or retrieval.load_or_build_retriever()
    configs = configs or DEFAULT_ABLATION_CONFIGS
    results, summary = {}, []
    for cfg in configs:
        name = cfg.pop("name")
        print(f"[ablation] 运行配置: {name} ({cfg})")
        res = eval_recall(retriever, queries, top_k=top_k, **cfg)
        results[name] = res
        summary.append((name, res["mean_recall"]))
    best = max((s for s in summary if s[1] is not None),
               key=lambda kv: kv[1], default=(None, None))
    payload = {
        "top_k": top_k,
        "num_queries": len(queries),
        "results": results,
        "best_config": {"name": best[0], "mean_recall": best[1]},
    }
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(out_file).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    # 打印对比表
    print("\n===== 消融实验结果 (Recall@{}) =====".format(top_k))
    for name, recall in sorted(summary, key=lambda kv: (kv[1] is None, -(kv[1] or 0))):
        bar = "#" * int(round((recall or 0) * 40))
        print(f"  {name:<18} {recall if recall is not None else 'N/A':>8}  {bar}")
    print(f"最优配置: {best[0]} (mean_recall={best[1]})")
    print(f"结果已保存 -> {out_file}")
    return payload


# ==========================================================================
# 命令行
# ==========================================================================
def run_cli(eval_file=DEFAULT_EVAL_FILE, mode: str = "all", top_k: int = 5,
            judge: str = "keyword", with_qa: bool = False) -> None:
    """
    评估 CLI 入口（main.py 的 eval 子命令转发到这里）。

    Args:
        mode: recall / qa / hallucination / ablation / all。
        with_qa: all 模式下是否追加问答准确率与幻觉率（需调用 LLM，成本高）。
    """
    ensure_dirs()
    if mode == "ablation":  # 消融只需评估集的 question 字段
        queries = load_eval_file(eval_file)
        run_ablation(queries, top_k=top_k)
        return

    queries = load_eval_file(eval_file)
    if mode == "recall":
        retriever = retrieval.load_or_build_retriever()
        result = eval_recall(retriever, queries, top_k=top_k)
        print(json.dumps({k: v for k, v in result.items() if k != "details"},
                         ensure_ascii=False, indent=2))
    elif mode == "qa":
        result = eval_qa(queries, judge=judge, top_k=top_k)
        print(json.dumps({k: v for k, v in result.items() if k != "details"},
                         ensure_ascii=False, indent=2))
    elif mode == "hallucination":
        result = eval_hallucination(queries, judge=judge, top_k=top_k)
        print(json.dumps({k: v for k, v in result.items() if k != "details"},
                         ensure_ascii=False, indent=2))
    else:  # all
        retriever = retrieval.load_or_build_retriever()
        recall_res = eval_recall(retriever, queries, top_k=top_k)
        print(f"[recall] mean_recall@{top_k} = {recall_res['mean_recall']}")
        run_ablation(queries, retriever=retriever, top_k=top_k)
        if with_qa:
            rag = rag_pipeline.RAGPipeline(retriever=retriever)
            qa_res = eval_qa(queries, rag=rag, judge=judge, top_k=top_k)
            print(f"[qa] accuracy({judge}) = {qa_res['accuracy']}")
            hal_res = eval_hallucination(queries, rag=rag, judge=judge, top_k=top_k)
            print(f"[hallucination] rate({judge}) = {hal_res['hallucination_rate']}")
        else:
            print("[提示] --with-qa 可追加问答准确率与幻觉率评估（需要调用 GLM-5.3）")


if __name__ == "__main__":
    import argparse
    setup_console()
    parser = argparse.ArgumentParser(description="RAG 评估与消融实验")
    parser.add_argument("--mode", default="all",
                        choices=["recall", "qa", "hallucination", "ablation", "all"])
    parser.add_argument("--eval-file", default=str(DEFAULT_EVAL_FILE))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--judge", default="keyword", choices=["keyword", "llm"])
    parser.add_argument("--with-qa", action="store_true")
    args = parser.parse_args()
    run_cli(eval_file=args.eval_file, mode=args.mode, top_k=args.top_k,
            judge=args.judge, with_qa=args.with_qa)
