# -*- coding: utf-8 -*-
"""
项目入口 demo（命令行）。

子命令
------
    parse    批量解析 papers/ 下的 PDF -> data/parsed/*.json
    chunk    多粒度分块 -> data/chunks/chunks.json
    index    构建混合检索索引（BM25 + FAISS + embedding-3 向量）
    ask      单次提问（--source 过滤 / --top-k 名额 / --summary 生成摘要）
    compare  多篇文献摘要横向对比: python -m src.main compare --source a.pdf --source b.pdf
    demo     端到端演示：解析 -> 分块 -> 索引 -> 提问 -> 跨论文分析
    analyze  跨论文分析：参考文献 / 引用网络图 / 指标对比
    eval     评估与消融: python -m src.main eval --mode ablation

典型流程
--------
    1. 把论文 PDF 放入 papers/
    2. 在 .env 填写 ZHIPUAI_API_KEY
    3. python -m src.main index        # 自动完成 parse + chunk + index
    4. python -m src.main ask "..."
    5. python -m src.main analyze
    6. python -m src.main eval --mode ablation

也支持以脚本方式运行: python src/main.py ...
"""

import argparse
import sys
from pathlib import Path

# 允许 `python src/main.py` 直接运行（此时无包上下文，把项目根加入 sys.path）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import chunking, evaluator, paper_analysis, pdf_parser, rag_pipeline, retrieval, summarizer
from src.config import (
    EXPERIMENTS_DIR, INDEX_DIR, PARSED_DIR, PAPERS_DIR, QA_LOG_FILE, ensure_dirs,
    setup_console,
)


# ==========================================================================
# 阶段性步骤（供各子命令与 demo 复用）
# ==========================================================================
def step_parse(force: bool = False) -> list:
    """
    步骤1：增量解析 PDF——只解析 papers/ 中新增（尚无解析结果）的文件，
    已解析的自动跳过；force=True 强制全部重解析（解析规则升级后使用）。
    """
    print("[1/4] 增量解析 papers/ 下的 PDF（已有结果的跳过，新增的才解析）...")
    return pdf_parser.batch_parse(PAPERS_DIR, PARSED_DIR, overwrite=force)


def step_index():
    """步骤2+3：分块与索引构建（已有索引则直接加载）。"""
    if (INDEX_DIR / "faiss.index").exists() and (INDEX_DIR / "meta.json").exists():
        print(f"[2-3/4] 已存在索引 {INDEX_DIR}，直接加载。")
        return retrieval.HybridRetriever.load(INDEX_DIR)
    print("[2-3/4] 分块并构建混合检索索引（含 embedding-3 向量化）...")
    return retrieval.build_index()


def _print_answer(res: dict, warn_lines=None, tail_note=None) -> None:
    """统一格式化打印问答结果（零召回警告头 + 答案 + 溯源 + 尾注提示）。"""
    print("\n" + "=" * 70)
    # 多文档零召回时的醒目警告头（⚠️）
    for line in (warn_lines or []):
        print(line)
    if warn_lines:
        print("-" * 70)
    print("【答案】")
    print(res["answer"])
    if res["insufficient"]:
        print(">> 提示：模型判定文献证据不足，可换一种问法或补充相关论文。")
    if res["invalid_markers"]:
        print(f">> 警告：答案引用了不存在的编号 {res['invalid_markers']}（潜在幻觉信号）。")
    if res["citations"]:
        print("\n【引用溯源】")
        for c in res["citations"]:
            print(f"  [{c['marker']}] {c.get('source') or '未知文件'} | "
                  f"《{c['paper_title']}》 "
                  f"{rag_pipeline.section_zh(c['section'])}部分 第{c['page']}页 "
                  f"(chunk: {c['chunk_id']})")
    # 文献覆盖统计：多文档均衡召回时展示每篇入选块数
    counts = res.get("per_source_counts") or {}
    if len(counts) >= 1 and res.get("balanced"):
        print("\n【文献覆盖】")
        print("  " + "  ".join(f"{s}×{n}块" for s, n in counts.items()))
    if tail_note:
        print(f"\n{tail_note}")
    print("=" * 70)


# ==========================================================================
# 各子命令实现
# ==========================================================================
def cmd_parse(args) -> None:
    """parse 子命令：批量解析 PDF（默认增量，--force 强制重解析）。"""
    results = pdf_parser.batch_parse(PAPERS_DIR, PARSED_DIR, overwrite=args.force)
    print(f"\n本次解析 {len(results)} 篇。JSON 位于: {PARSED_DIR}")


def cmd_chunk(args) -> None:
    """chunk 子命令：多粒度分块。"""
    chunks = chunking.build_all_chunks(PARSED_DIR, chunking.CHUNKS_FILE)
    print(f"\n分块完成，共 {len(chunks)} 块。文件: {chunking.CHUNKS_FILE}")


def cmd_index(args) -> None:
    """index 子命令：确保解析 -> 分块 -> 构建/重建索引。"""
    step_parse()
    retriever = retrieval.build_index()
    print(f"\n索引构建完成: {INDEX_DIR}（共 {len(retriever.chunks)} 个分块）")


# 多文档联合查询的使用建议（终端提示文案）
MULTI_SOURCE_TIP = (
    "[提示] 检测到多文档联合查询: 已启用按文献均衡召回（每篇保底配额，"
    "短文档不会被长文档完全挤出）。如需更深入的单篇细节，建议分开查询，"
    "或先 --summary 生成各篇摘要，再用 compare 完成对比。"
)


def cmd_add(args) -> None:
    """
    add 子命令：增量导入 papers/ 新增 PDF —— 只解析新文件、只对
    新增/内容变更的分块调用向量化 API，已有分块复用向量缓存，
    不需要全量重建向量库。
    """
    ensure_dirs()
    print("[增量导入] 解析 papers/ 中新增的 PDF ...")
    step_parse()
    print("[增量导入] 分块并增量更新向量索引 ...")
    retriever = retrieval.build_index()
    print(f"\n增量导入完成: {INDEX_DIR}（共 {len(retriever.chunks)} 个分块，"
          f"已有分块复用缓存，未重复消耗 API 额度）")


def cmd_ask(args) -> None:
    """
    ask 子命令：单次问答 / --summary 结构化摘要生成。

    - 默认: 检索 + GLM-5.3 受约束问答（支持 --source 过滤、--top-k 名额）；
    - --summary: 针对单个 --source 指定的 PDF 预生成固定模板摘要并保存
      Markdown（预生成笔记优先，RAG 问答仅用于深挖细节）。
    """
    ensure_dirs()
    # ---- --summary 参数校验（无需加载索引，先拦截参数错误） ----
    if args.summary:
        if not args.source:
            print("[错误] --summary 需要配合 --source 指定一篇PDF，例如:\n"
                  "    python -m src.main ask --source paper3.pdf --summary")
            return
        if len(args.source) > 1:
            print("[错误] --summary 仅支持单个 --source（一次只为一篇文献生成摘要）。\n"
                  "多篇对比请使用 compare 子命令，例如:\n"
                  f"    python -m src.main compare {' '.join('--source ' + s for s in args.source)}")
            return
    if not args.summary and not args.question:
        print('[错误] 请提供问题，例如: python -m src.main ask "研究方法是什么"')
        return

    retriever = retrieval.load_or_build_retriever()
    rag = rag_pipeline.RAGPipeline(retriever=retriever)

    # ---- 摘要生成分支 ----
    if args.summary:
        if args.question:
            print(f"[提示] --summary 模式忽略问题参数: {args.question}")
        try:
            res = summarizer.generate_summary(args.source[0], rag)
        except ValueError as e:
            print(f"[错误] {e}")
            return
        print("\n" + res["markdown"])
        print(f"\n摘要已保存: {res['path']}")
        return

    # ---- 常规问答分支 ----
    metadata_filter = {"source": args.source} if args.source else None
    multi = bool(args.source and len(args.source) >= 2)
    if multi:
        print(MULTI_SOURCE_TIP)
    try:
        res = rag.ask(args.question, top_k=args.top_k, mode=args.mode,
                      metadata_filter=metadata_filter,
                      rerank=(False if args.no_rerank else None))
    except KeyboardInterrupt:
        print("\n已取消")
        return
    except Exception as e:
        # 任务⑥：问答失败给出友好提示（含限流/额度场景），不打印裸堆栈
        print(f"\n[错误] 问答失败: {e}")
        return
    # 多 source 查询出现零召回文档时：输出头部⚠️警告 + 末尾对比建议
    warn_lines = summarizer.zero_recall_warn_lines(res, multi_source=multi)
    tail_note = summarizer.COMPARE_TIP if (multi and (res.get("unrecalled_sources") or [])) else None
    _print_answer(res, warn_lines=warn_lines, tail_note=tail_note)
    _warn_unrecalled(res)
    print(f"问答日志: logs/qa.log | logs/qa.jsonl | logs/retrieve.log")


def cmd_compare(args) -> None:
    """compare 子命令：多篇文献结构化摘要的横向对比（摘要缺失时自动补生成）。"""
    if not args.source or len(args.source) < 2:
        print("[错误] compare 至少需要两个 --source，例如:\n"
              "    python -m src.main compare --source paper1.pdf --source paper5.pdf")
        return
    retriever = retrieval.load_or_build_retriever()
    rag = rag_pipeline.RAGPipeline(retriever=retriever)
    try:
        res = summarizer.compare_summaries(args.source, rag, refresh=args.refresh)
    except ValueError as e:
        print(f"[错误] {e}")
        return
    print("\n" + res["markdown"])
    print(f"\n对比报告已保存: {res['path']}")
    if res["generated"]:
        print(f"[提示] 以下文献的摘要为本次新生成: {', '.join(res['generated'])}")


def _warn_unrecalled(res: dict) -> None:
    """
    零召回告警：指出哪些文献没有片段进入上下文，按原因给出建议。
    - not_in_index: 文件名不在索引中（拼写错误或未建索引）
    - not_recalled: 显式 source 过滤下被其他文档挤出 top-k
    - no_relevant:  全库均衡召回下，该文献未检索到与问题相关的片段
    """
    unrecalled = res.get("unrecalled_sources") or []
    if not unrecalled:
        return
    print("[警告] 以下 source 未召回任何片段:")
    for item in unrecalled:
        if item["reason"] == "not_in_index":
            print(f"  - {item['source']}（不在当前索引中: 请确认文件名，"
                  f"或先运行 python -m src.main index）")
        elif item["reason"] == "no_relevant":
            print(f"  - {item['source']}（未检索到与该问题相关的片段: "
                  f"该文献可能与问题无关）")
        else:
            print(f"  - {item['source']}（已索引但相关片段不足，被其他文档"
                  f"挤出 top-k: 可增大 --top-k 缓解，或单独查询该文档）")


def cmd_demo(args) -> None:
    """demo 子命令：端到端演示（解析 -> 索引 -> 问答 -> 分析）。"""
    ensure_dirs()
    print("=" * 70)
    print("学术文献 RAG 研读助手 —— 端到端演示")
    print("=" * 70)

    # 步骤 1：解析
    step_parse()
    if not any(PARSED_DIR.glob("*.json")):
        print("\n[中止] papers/ 下没有 PDF 且无解析结果。请先放入论文 PDF。")
        return

    # 步骤 2-3：分块 + 索引
    retriever = step_index()

    # 步骤 4：问答演示
    print("\n[4/4] RAG 问答演示 ...")
    rag = rag_pipeline.RAGPipeline(retriever=retriever)
    question = args.question or "请概括文献库中各论文的核心方法与主要实验结论，并标注来源"
    print(f"问题: {question}")
    try:
        res = rag.ask(question)
        _print_answer(res)
    except Exception as e:
        print(f"[问答失败] {e}")

    # 附加：跨论文分析（不阻塞主流程）
    print("\n[附加] 跨论文分析（参考文献 / 引用网络 / 指标对比）...")
    try:
        paper_analysis.analyze_all(PARSED_DIR)
    except Exception as e:
        print(f"[分析失败] {e}")

    print("\n演示结束。后续可运行:")
    print("  python -m src.main eval --mode ablation   # 检索消融实验")
    print("  python -m src.main ask \"你的问题\"          # 继续提问")


def cmd_analyze(args) -> None:
    """analyze 子命令：跨论文指标对比 / 参考文献提取 / 引用网络图。"""
    paper_analysis.analyze_all(PARSED_DIR)


def cmd_eval(args) -> None:
    """eval 子命令：召回率 / 准确率 / 幻觉率 / 消融。"""
    evaluator.run_cli(eval_file=args.eval_file, mode=args.mode,
                      top_k=args.top_k, judge=args.judge, with_qa=args.with_qa)


# ==========================================================================
# CLI 定义
# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="src.main",
        description="学术文献 RAG 研读助手（LlamaIndex + FAISS + BM25 + GLM-5.3）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("典型流程")[0] if __doc__ else None,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_parse = sub.add_parser("parse", help="批量解析 papers/ 下的 PDF（默认增量）")
    p_parse.add_argument("--force", action="store_true",
                         help="忽略已有解析结果，强制全部重新解析")
    p_parse.set_defaults(func=cmd_parse)
    sub.add_parser("chunk", help="对解析结果做多粒度分块").set_defaults(func=cmd_chunk)
    sub.add_parser("index", help="构建混合检索索引（BM25 + FAISS）").set_defaults(func=cmd_index)
    sub.add_parser("add", help="增量导入 papers/ 新增PDF（仅新增分块向量化，不全量重建）").set_defaults(func=cmd_add)

    p_ask = sub.add_parser("ask", help="单次提问（带引用溯源）；--summary 生成结构化摘要")
    p_ask.add_argument("question", nargs="?", default=None,
                       help="要提问的问题（--summary 模式下可省略）。未指定 --source 时"
                            "自动启用按文献均衡召回，覆盖 papers/ 全部文献并逐篇解答；"
                            "单独提问某篇请加 --source")
    p_ask.add_argument("--mode", default="hybrid", choices=["bm25", "dense", "hybrid"],
                       help="检索模式（默认 hybrid）")
    p_ask.add_argument("--top-k", type=int, default=12,
                       help="检索返回并送入上下文的分块数（默认12；多文档联合查询"
                            "时可调大，缓解短文档被长文档挤出召回）")
    p_ask.add_argument("--source", action="append", default=None,
                       help="元数据过滤：只在指定PDF内检索（原始文件名，"
                            "可重复传入多个，如 --source paper1.pdf）")
    p_ask.add_argument("--summary", action="store_true",
                       help="为单个 --source 指定的PDF生成固定模板结构化摘要，"
                            "保存到 literature_lib/summary/<主文件名>.md")
    p_ask.add_argument("--no-rerank", action="store_true",
                       help="关闭 Reranker 重排序（默认按 RERANK_ENABLED 配置，"
                            "开启时先取候选重排再取 top_k）")
    p_ask.set_defaults(func=cmd_ask)

    p_cmp = sub.add_parser("compare", help="多篇文献结构化摘要横向对比（缺失的摘要自动补生成）")
    p_cmp.add_argument("--source", action="append", required=True,
                       help="参与对比的PDF文件名（至少两个，可重复传入）")
    p_cmp.add_argument("--refresh", action="store_true",
                       help="忽略已保存的摘要，全部重新生成后再对比")
    p_cmp.set_defaults(func=cmd_compare)

    p_demo = sub.add_parser("demo", help="端到端演示")
    p_demo.add_argument("--question", default=None, help="演示问题（默认使用内置问题）")
    p_demo.set_defaults(func=cmd_demo)

    sub.add_parser("analyze", help="跨论文分析（引用网络/指标对比）").set_defaults(func=cmd_analyze)

    p_eval = sub.add_parser("eval", help="评估与消融实验")
    p_eval.add_argument("--mode", default="all",
                        choices=["recall", "qa", "hallucination", "ablation", "all"])
    p_eval.add_argument("--eval-file", default=str(EXPERIMENTS_DIR / "eval_set.json"))
    p_eval.add_argument("--top-k", type=int, default=5)
    p_eval.add_argument("--judge", default="keyword", choices=["keyword", "llm"],
                        help="问答准确率判定方式：keyword 离线 / llm 调用GLM-5.3判卷")
    p_eval.add_argument("--with-qa", action="store_true",
                        help="all 模式下追加问答准确率与幻觉率评估（消耗API额度）")
    p_eval.set_defaults(func=cmd_eval)
    return parser


def main() -> None:
    setup_console()
    ensure_dirs()
    args = build_parser().parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        # 任务⑥：Ctrl+C 干净退出，不打印堆栈
        print("\n已退出")
    except Exception as e:
        # 任务⑥：顶层异常兜底——任何子命令失败都给友好提示而非裸崩溃
        print(f"\n[错误] {type(e).__name__}: {e}")
        print("提示: 若为API相关错误，请检查 .env 密钥、网络或平台限流/额度后重试；"
              "更多细节可在控制台日志中查看。")


if __name__ == "__main__":
    main()
