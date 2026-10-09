# -*- coding: utf-8 -*-
"""
CMD 终端问答脚本 —— 复用项目内部 ask 函数（RAGPipeline.ask），与
Web 前端/CLI 主入口共享同一套检索、重排、防幻觉与日志逻辑。

用法
----
    python cli_ask.py                                  # 全部文献（均衡召回）
    python cli_ask.py --source paper1.pdf              # 仅针对某一篇提问
    python cli_ask.py --top-k 12 --no-rerank           # 关闭重排序

交互
----
    问题> 输入问题回车即提问
    输入 quit / exit / q 退出
每一条问答（问题/检索片段/回答/参数）自动写入 output/qa_log.jsonl 与
output/qa_log.txt。
"""

import argparse
import sys
from pathlib import Path

# 允许 `python cli_ask.py` 直接运行
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import rag_pipeline, summarizer
from src.config import ensure_dirs, setup_console
# 复用 CLI 主入口的回答打印与零召回告警格式（不重复实现展示逻辑）
from src.main import _print_answer, _warn_unrecalled

EXIT_WORDS = {"quit", "exit", "q", "退出"}


def main() -> None:
    setup_console()  # Windows 控制台 UTF-8
    ensure_dirs()

    parser = argparse.ArgumentParser(
        description="CMD 终端问答（复用项目内部 ask 函数；输入 quit 退出）")
    parser.add_argument("--source", default=None,
                        help="仅针对该 PDF 作答（文件名，缺省=全部文献均衡召回逐篇解答）")
    parser.add_argument("--top-k", type=int, default=12,
                        help="最终送入大模型的分块数（默认12）")
    parser.add_argument("--mode", default="hybrid",
                        choices=["bm25", "dense", "hybrid"], help="检索模式")
    parser.add_argument("--no-rerank", action="store_true",
                        help="关闭 Reranker 重排序（默认按 RERANK_ENABLED 配置）")
    args = parser.parse_args()

    print("正在加载索引（首次可能较慢）...")
    rag = rag_pipeline.RAGPipeline()  # 复用同一问答链路（检索/重排/溯源/日志）
    metadata_filter = {"source": args.source} if args.source else None

    print("=" * 70)
    print("学术文献 RAG 终端问答")
    print(f"  范围: {args.source or '全部文献（均衡召回，逐篇解答）'} | "
          f"top_k={args.top_k} | 模式={args.mode} | "
          f"重排={'关闭' if args.no_rerank else '跟随配置'}")
    print("  输入问题开始提问，输入 quit / exit / q 退出")
    print("=" * 70)

    while True:
        try:
            question = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not question:
            continue
        if question.lower() in EXIT_WORDS:
            print("再见！")
            break
        try:
            res = rag.ask(question, top_k=args.top_k, mode=args.mode,
                          metadata_filter=metadata_filter,
                          rerank=(False if args.no_rerank else None))
            _print_answer(res)       # 答案 + 引用溯源 + 文献覆盖（复用主入口格式）
            _warn_unrecalled(res)    # 零召回告警
            print(f"日志: logs/qa.log | logs/qa.jsonl | logs/retrieve.log")
        except Exception as e:
            print(f"[错误] {e}")


if __name__ == "__main__":
    main()
