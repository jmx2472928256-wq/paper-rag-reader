# -*- coding: utf-8 -*-
"""
批量文献处理脚本 —— 一次运行完成多篇文献的摘要生成与问答。

设计背景：预生成笔记优先。先批量生成各篇结构化摘要作为日常阅读材料，
RAG 问答仅用于深挖细节，避免逐次实时检索的召回不稳定问题。

任务文件（JSON，默认 literature_lib/batch_tasks.json）
------------------------------------------------------
{
  "save_markdown": true,                       // 可选：问答结果保存为 Markdown
  "tasks": [
    ["paper1.pdf", "这篇论文的研究方法是什么？"],   // 普通问答任务: (pdf, 问题)
    ["paper5.pdf", "**summary**"],                 // 摘要生成任务: 问题固定为 **summary**
    {"pdf": "paper3.pdf", "question": "主要结论是什么？",
     "save_md": false}                             // 对象写法（save_md 可覆盖全局开关）
  ]
}

用法
----
    python batch_run.py                                # 默认读取 literature_lib/batch_tasks.json
    python batch_run.py --tasks my_tasks.json --save-markdown

输出
----
    摘要任务 -> literature_lib/summary/<pdf主文件名>.md（同时控制台打印）
    问答任务 -> 控制台打印；save_markdown 开启时另存 literature_lib/answers/<pdf>_<序号>.md
"""

import argparse
import json
import sys
from pathlib import Path

# 允许直接以 `python batch_run.py` 运行
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import rag_pipeline, retrieval, summarizer
from src.config import ANSWERS_DIR, ensure_dirs, setup_console
from src.main import _warn_unrecalled

SUMMARY_MARKER = "**summary**"  # 摘要任务的问题标记（需求约定）


def _iter_tasks(data: dict):
    """
    统一解析两种任务写法 -> 产生 (序号, pdf, question, save_md或None)。

    - 数组写法: ["paper1.pdf", "问题文本"]
    - 对象写法: {"pdf": "...", "question": "...", "save_md": true|false}
    """
    tasks = data.get("tasks") or []
    for i, t in enumerate(tasks, 1):
        if isinstance(t, dict):
            yield i, t.get("pdf", ""), t.get("question", ""), t.get("save_md")
        elif isinstance(t, (list, tuple)) and len(t) >= 2:
            yield i, t[0], t[1], None
        else:
            yield i, None, None, None  # 非法条目，交由主循环报错


def _write_template(path: Path) -> None:
    """任务文件不存在时生成一份可直接编辑的模板。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    template = {
        "save_markdown": False,
        "tasks": [
            ["paper1.pdf", "这篇论文的研究方法是什么？"],
            ["paper1.pdf", SUMMARY_MARKER],
        ],
    }
    path.write_text(json.dumps(template, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"[batch_run] 任务文件不存在，已生成模板: {path}\n"
          f"            请编辑 tasks 后重新运行（question 为 {SUMMARY_MARKER} 即摘要任务）。")


def _print_batch_answer(res: dict) -> None:
    """批处理下的精简问答展示（完整溯源写入 Markdown / qa_log）。"""
    print("【答案】")
    print(res["answer"])
    if res.get("invalid_markers"):
        print(f">> 警告: 答案引用了不存在的编号 {res['invalid_markers']}（潜在幻觉信号）")
    if res.get("citations"):
        print("【引用来源】")
        for c in res["citations"]:
            print(f"  [{c['marker']}] {c.get('source', '?')} | "
                  f"《{c.get('paper_title', '')}》 {c.get('section', '')} "
                  f"第{c.get('page', '?')}页")


def main() -> None:
    setup_console()
    parser = argparse.ArgumentParser(
        description="批量文献摘要生成与问答（任务文件见脚本 docstring）")
    parser.add_argument("--tasks", default="literature_lib/batch_tasks.json",
                        help="任务文件路径（JSON，默认 literature_lib/batch_tasks.json）")
    parser.add_argument("--save-markdown", dest="save_markdown", action="store_true",
                        default=None,
                        help="保存问答结果为Markdown（覆盖任务文件中的 save_markdown 设置）")
    args = parser.parse_args()

    task_file = Path(args.tasks)
    if not task_file.exists():
        _write_template(task_file)
        return

    cfg = json.loads(task_file.read_text(encoding="utf-8"))
    # 保存开关优先级: 命令行 > 任务文件 > 默认 False
    save_global = (args.save_markdown if args.save_markdown is not None
                   else bool(cfg.get("save_markdown", False)))

    ensure_dirs()
    retriever = retrieval.load_or_build_retriever()
    rag = rag_pipeline.RAGPipeline(retriever=retriever)
    indexed = {c.get("source") for c in retriever.chunks}

    n_ok = n_fail = 0
    for idx, pdf, question, save_md in _iter_tasks(cfg):
        print("\n" + "=" * 70)
        is_summary = bool(question) and question.strip() == SUMMARY_MARKER
        print(f"[任务{idx:02d}] {pdf or '(缺少pdf)'} | "
              f"{'生成结构化摘要' if is_summary else (question or '(缺少question)')[:40]}")
        try:
            if not pdf or not question:
                raise ValueError("任务格式错误，需要 (pdf, question) 数组或对象写法")
            if pdf not in indexed:
                raise ValueError(f"{pdf} 不在索引中（可用: "
                                 f"{sorted(x for x in indexed if x)}）")

            if is_summary:
                # ---- 摘要生成任务: 复用 ask --summary 逻辑，自动输出 md ----
                res = summarizer.generate_summary(pdf, rag)
                print(res["markdown"])
                print(f"[完成] 摘要已保存: {res['path']}")
            else:
                # ---- 普通问答任务: 原有 ask 逻辑（单source过滤）----
                res = rag.ask(question, metadata_filter={"source": pdf}, save_log=True)
                _print_batch_answer(res)
                _warn_unrecalled(res)  # 零召回告警（与 ask 命令一致）
                should_save = save_global if save_md is None else bool(save_md)
                if should_save:
                    md = summarizer.build_qa_markdown(
                        res, metadata_filter={"source": pdf})
                    out = ANSWERS_DIR / f"{Path(pdf).stem}_{idx:02d}.md"
                    out.write_text(md, encoding="utf-8")
                    print(f"[完成] 问答已保存: {out}")
            n_ok += 1
        except Exception as e:  # 单任务失败不中断批处理
            n_fail += 1
            print(f"[失败] {e}")

    print("\n" + "=" * 70)
    print(f"批量任务结束: 成功 {n_ok}, 失败 {n_fail}")
    print(f"摘要目录: literature_lib/summary/ | 问答Markdown: literature_lib/answers/")


if __name__ == "__main__":
    main()
