# -*- coding: utf-8 -*-
"""
跨论文分析模块。

功能
----
1. 参考文献提取：从解析结果的 references 章节抽取参考文献条目
   （支持 [n] 编号式与作者-年份式两种排版），保存 JSON；
2. 引用网络图：将每篇论文的参考文献与文献库内其他论文做标题模糊匹配，
   用 networkx 构建"谁引用了谁"的有向图，matplotlib 绘制并保存 PNG；
3. 跨论文指标对比：从实验/结果章节用正则抽取常见指标数值
   （accuracy / F1 / BLEU / ROUGE / recall / mAP / AUC ... 支持中文指标名），
   汇总为对比表保存 CSV。

输出（全部写入 output/）
------------------------
    references.json          每篇论文的参考文献列表
    citation_edges.csv       文献库内引用边（source -> target）
    citation_network.png     引用网络图
    metrics_raw.csv          指标抽取明细
    metrics_comparison.csv   论文 x 指标 对比透视表

用法
----
    python -m src.paper_analysis
"""

import csv
import difflib
import json
import re
from pathlib import Path

import networkx as nx
import pandas as pd

try:
    from .config import OUTPUT_DIR, PARSED_DIR, ensure_dirs, setup_console
except ImportError:  # 以脚本方式直接运行
    from config import OUTPUT_DIR, PARSED_DIR, ensure_dirs, setup_console

# 引用网络绘图使用无界面后端（服务器/无显示环境也能出图）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文标签字体（Windows 自带微软雅黑/黑体；其他系统按顺序回退）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC",
                                   "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ==========================================================================
# 1. 参考文献提取
# ==========================================================================
def _find_references_text(parsed: dict) -> str:
    """定位解析结果中的参考文献章节正文（优先规范名，其次标题含 reference）。"""
    for sec in parsed.get("sections", []):
        if sec.get("name") == "references":
            return sec.get("text", "")
    for sec in parsed.get("sections", []):
        if "reference" in sec.get("heading", "").lower() or "参考文献" in sec.get("heading", ""):
            return sec.get("text", "")
    return ""


def _split_numbered(ref_text: str) -> list:
    r"""
    切分 [1] [2] 编号式参考文献：PDF 断行先折叠为空格，再在 "[数字]"
    前切开（(?=\[\d+\]) 为零宽断言，保留编号在条目内）。
    """
    flat = re.sub(r"\s+", " ", ref_text)
    entries = re.split(r"(?=\[\d{1,3}\])", flat)
    cleaned = []
    for e in entries:
        e = re.sub(r"^\[\d{1,3}\]\s*", "", e).strip()
        # 过滤过短碎片（页码/乱入 token）与明显非文献内容
        if len(e) >= 15:
            cleaned.append(e)
    return cleaned


_AUTHOR_YEAR_START = re.compile(r"^[A-Z][A-Za-z\-'\".]+\s*(?:,|and|&|[A-Z])")
_YEAR_END = re.compile(r"\d{4}\s*[a-z]?(?:[.,;)]|\)|$)")


def _split_author_year(ref_text: str) -> list:
    """
    切分作者-年份式（APA/MLA 等）参考文献：逐行扫描，当出现"大写开头
    作者模式"且上一条已以年份/句点收尾时开新条目。启发式，允许一定误差。
    """
    entries, buf = [], ""
    for raw in ref_text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        starts_new = bool(_AUTHOR_YEAR_START.match(line)) and (
            not buf or bool(_YEAR_END.search(buf[-15:]))
        )
        if starts_new and buf:
            entries.append(buf.strip())
            buf = line
        else:
            buf = f"{buf} {line}".strip()
    if len(buf) >= 15:
        entries.append(buf.strip())
    return [e for e in entries if len(e) >= 15]


def extract_references(parsed: dict) -> list:
    """
    提取单篇论文的参考文献条目列表（优先编号式，失败回退作者-年份式）。
    """
    ref_text = _find_references_text(parsed)
    if not ref_text:
        return []
    entries = _split_numbered(ref_text)
    # 编号式只切出1条时可能 indeed 只有1条文献，也可能编号式失效；
    # 为稳妥起见 >=2 条直接采用，==1 条时尝试作者-年份式，取结果更多者
    if len(entries) < 2:
        alt = _split_author_year(ref_text)
        entries = alt if len(alt) > len(entries) else entries
    return entries


# ==========================================================================
# 2. 引用网络
# ==========================================================================
def _norm_title(s: str) -> str:
    """标题归一化：小写 + 仅保留字母数字与中文，供模糊匹配。"""
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", s.lower())


def match_reference_to_corpus(ref: str, corpus_titles: dict, min_ratio: float = 0.75):
    """
    将一条参考文献字符串匹配到文献库内的某篇论文。

    Args:
        ref: 参考文献条目原文。
        corpus_titles: {paper_id: 标题}。
        min_ratio: difflib 相似度阈值。

    Returns:
        匹配到的 paper_id；无匹配返回 None。

    匹配策略（从严到宽）：
    1. 归一化后互相包含（参考文献中标题常完整出现）；
    2. difflib 序列相似度 >= 0.75（容忍 OCR/断行误差）。
    标题过短（<15字符）不参与匹配，避免误命中。
    """
    ref_norm = _norm_title(ref)
    if len(ref_norm) < 10:
        return None
    best_id, best_ratio = None, 0.0
    for pid, title in corpus_titles.items():
        t_norm = _norm_title(title)
        if len(t_norm) < 15:  # 太短的标题易误匹配
            continue
        if t_norm in ref_norm or ref_norm in t_norm:
            return pid  # 完整包含即视为命中
        ratio = difflib.SequenceMatcher(None, ref_norm, t_norm).ratio()
        if ratio > best_ratio:
            best_id, best_ratio = pid, ratio
    return best_id if best_ratio >= min_ratio else None


def build_citation_graph(parsed_list: list) -> nx.DiGraph:
    """
    构建文献库内引用网络：
    节点 = 论文，边 = "引用论文 -> 被引论文"（仅限库内可匹配到的引用）。
    边属性 edge_type="internal"；同时把库外参考文献数量记入节点属性。
    """
    corpus_titles = {p["paper_id"]: p.get("title", "") for p in parsed_list}
    graph = nx.DiGraph()
    for p in parsed_list:
        graph.add_node(p["paper_id"], title=p.get("title", ""), filename=p.get("filename", ""))

    for p in parsed_list:
        refs = extract_references(p)
        external = 0
        for ref in refs:
            target = match_reference_to_corpus(ref, corpus_titles)
            if target and target != p["paper_id"]:
                # 自引忽略；库内引用建边
                if graph.has_edge(p["paper_id"], target):
                    graph[p["paper_id"]][target]["weight"] += 1
                else:
                    graph.add_edge(p["paper_id"], target, weight=1, edge_type="internal")
            elif target is None:
                external += 1
        graph.nodes[p["paper_id"]]["num_references"] = len(refs)
        graph.nodes[p["paper_id"]]["num_external_refs"] = external
    return graph


def plot_citation_network(graph: nx.DiGraph, out_path) -> Path:
    """
    绘制引用网络图并保存 PNG。节点大小 = 被引次数（入度+1），
    布局用 spring layout（固定随机种子保证可复现）。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if graph.number_of_nodes() == 0:
        print("[paper_analysis] 引用网络为空，跳过绘图。")
        return out_path

    n = graph.number_of_nodes()
    fig_w = max(8.0, min(16.0, 2.0 * n ** 0.5 + 4))
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.75))
    pos = nx.spring_layout(graph, seed=42, k=2.0 / max(1, n ** 0.5))

    # 节点：按被引次数缩放
    in_degs = dict(graph.in_degree())
    sizes = [600 + 900 * in_degs.get(node, 0) for node in graph.nodes()]
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=sizes,
                           node_color="#7fb3d5", edgecolors="#2c3e50", alpha=0.9)
    nx.draw_networkx_edges(graph, pos, ax=ax, arrowstyle="-|>",
                           arrowsize=14, edge_color="#95a5a6",
                           connectionstyle="arc3,rad=0.12", alpha=0.7)
    # 标签：标题截断到 16 字符
    labels = {}
    for node, attrs in graph.nodes(data=True):
        t = attrs.get("title", node)
        labels[node] = (t[:16] + "…") if len(t) > 16 else (t or node)
    nx.draw_networkx_labels(graph, pos, labels, ax=ax, font_size=9)

    ax.set_title("文献库引用网络（箭头: 引用 -> 被引）", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[paper_analysis] 引用网络图已保存 -> {out_path}")
    return out_path


# ==========================================================================
# 3. 跨论文指标对比
# ==========================================================================
# 指标名 -> 规范名（英文常见缩写 + 中文）
_METRIC_ALIAS = {
    "accuracy": "accuracy", "acc": "accuracy", "准确率": "accuracy",
    "precision": "precision", "精确率": "precision", "精度": "precision",
    "recall": "recall", "召回率": "recall",
    "f1": "f1", "f1score": "f1", "f1值": "f1", "f1分数": "f1",
    "bleu": "bleu", "rouge": "rouge", "map": "map", "mrr": "mrr",
    "auc": "auc", "em": "em", "wer": "wer",
}
# 形如 "accuracy of 92.3%" / "F1 score: 88.5" / "准确率达到 91.2%"
_METRIC_VALUE_RE = re.compile(
    r"(accuracy|acc(?:uracy)?|precision|recall|f1(?:[-\s]?score)?|bleu|"
    r"rouge(?:-[123l])?|map|mrr|auc|em|wer|准确率|精确率|精度|召回率|f1值|f1分数)"
    r"[^\d\-]{0,15}?"
    r"(\d{1,3}(?:\.\d+)?)\s*(%|％)?",
    re.IGNORECASE,
)


def extract_metrics(parsed: dict) -> list:
    """
    从实验/结果/方法章节抽取 (指标, 数值) 明细。

    过滤规则：带百分号的数值须落在 [0, 100]；未带百分号的数值
    需 <= 100（常见指标量纲），避免抓到年份、引用编号等噪声。
    """
    records = []
    for sec in parsed.get("sections", []):
        if sec.get("name") not in ("experiment", "results", "method", "abstract"):
            continue
        for m in _METRIC_VALUE_RE.finditer(sec.get("text", "")):
            raw_metric, value, pct = m.group(1), float(m.group(2)), m.group(3)
            metric = _METRIC_ALIAS.get(raw_metric.lower().replace(" ", "").replace("-", ""), raw_metric.lower())
            if pct and not (0.0 <= value <= 100.0):
                continue  # 百分比越界，明显是误抽
            if not pct and value > 100:
                continue
            records.append({
                "paper_id": parsed["paper_id"],
                "paper_title": parsed.get("title", ""),
                "section": sec["name"],
                "metric": metric,
                "value": value,
                "unit": "%" if pct else "",
                "context": m.group(0)[:80],
            })
    return records


def compare_metrics(parsed_list: list):
    """
    汇总所有论文的指标 -> 明细表 + 透视对比表（论文 x 指标，取最大值，
    即论文报告的最好成绩），保存 CSV 并返回透视表。
    """
    rows = []
    for p in parsed_list:
        rows.extend(extract_metrics(p))

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "metrics_raw.csv"
    pd.DataFrame(rows).to_csv(raw_path, index=False, encoding="utf-8-sig")

    if not rows:
        print("[paper_analysis] 未抽取到任何指标数值（论文可能是扫描版或指标名未覆盖）。")
        return None
    df = pd.DataFrame(rows)
    # 同一 (论文, 指标) 可能出现多次，取最大值作为论文报告的最佳成绩
    pivot = (df.groupby(["paper_title", "metric"])["value"].max()
               .unstack("metric").round(2))
    pivot_path = out_dir / "metrics_comparison.csv"
    pivot.to_csv(pivot_path, encoding="utf-8-sig")
    print(f"[paper_analysis] 指标明细 -> {raw_path}")
    print(f"[paper_analysis] 指标对比表 -> {pivot_path}")
    return pivot


# ==========================================================================
# 汇总入口
# ==========================================================================
def analyze_all(parsed_dir=PARSED_DIR, out_dir=OUTPUT_DIR):
    """
    端到端分析：参考文献 -> 引用网络（图+边表） -> 指标对比。
    """
    ensure_dirs()
    parsed_dir, out_dir = Path(parsed_dir), Path(out_dir)
    files = sorted(parsed_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError("data/parsed/ 为空，请先运行: python -m src.main parse")
    parsed_list = [json.loads(f.read_text(encoding="utf-8")) for f in files]

    # ---- 参考文献 ----
    refs_map = {p["paper_id"]: extract_references(p) for p in parsed_list}
    refs_path = out_dir / "references.json"
    refs_path.write_text(json.dumps(refs_map, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    for pid, refs in refs_map.items():
        print(f"  参考文献: {pid} -> {len(refs)} 条")
    print(f"[paper_analysis] 参考文献已保存 -> {refs_path}")

    # ---- 引用网络 ----
    graph = build_citation_graph(parsed_list)
    edges_path = out_dir / "citation_edges.csv"
    with open(edges_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["source_title", "target_title", "weight"])
        for u, v, d in graph.edges(data=True):
            writer.writerow([graph.nodes[u]["title"], graph.nodes[v]["title"],
                             d.get("weight", 1)])
    print(f"[paper_analysis] 引用边表 ({graph.number_of_edges()} 条) -> {edges_path}")
    plot_citation_network(graph, out_dir / "citation_network.png")

    # ---- 指标对比 ----
    compare_metrics(parsed_list)
    return {"references": refs_map, "graph": graph}


if __name__ == "__main__":
    setup_console()
    analyze_all()
