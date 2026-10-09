# -*- coding: utf-8 -*-
"""
多粒度分块模块。

功能
----
读取 data/parsed/ 下的解析 JSON，为每篇论文的每个章节生成三种粒度的文本块：

1. sentence  句子级（细粒度）：按句末标点切分后打包，目标约 260 字符，
              适合"某个指标/定义在哪"这类精确检索；
2. paragraph 段落级（中粒度）：滑窗切分，目标约 650 字符、重叠 120 字符，
              兼顾语义完整与检索精度，是问答主力粒度；
3. section   章节级（粗粒度）：整章保留（超长截断至约 2600 字符），
              用于需要大上下文的对比/总结类问题。

每个 chunk 记录：全局唯一 chunk_id、所属论文（id/标题）、规范章节名、
粒度、估算页码（由解析阶段的 page_spans 反查）、字符数等，便于检索结果
直接映射为「论文-章节-页码」的引用溯源信息。

若安装了 LlamaIndex，段落级切分优先使用其 SentenceSplitter（对中英文
句边界更稳健）；否则使用内置的滑窗+句边界对齐切分，两种路径行为一致。

输出：data/chunks/chunks.json（全部论文的 chunk 列表）

用法
----
    python -m src.chunking
"""

import json
import re
from bisect import bisect_right
from pathlib import Path

try:
    from .config import (
        CHUNK_MODE, CHUNK_OVERLAP_RATIO, CHUNKS_FILE, PARSED_DIR, ensure_dirs,
        setup_console,
    )
except ImportError:  # 以脚本方式直接运行
    from config import (
        CHUNK_MODE, CHUNK_OVERLAP_RATIO, CHUNKS_FILE, PARSED_DIR, ensure_dirs,
        setup_console,
    )

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# LlamaIndex 为可选增强：安装后段落级切分用 SentenceSplitter（token 感知）
try:
    from llama_index.core.node_parser import SentenceSplitter
    _HAS_LLAMA_INDEX = True
except Exception:
    _HAS_LLAMA_INDEX = False


# ==========================================================================
# 粒度配置（单位：字符。PDF 文本无可靠 token 计数，字符数是稳定代理）
# ==========================================================================
def _overlap_for(target_chars: int) -> int:
    """块间重叠 = chunk_size * CHUNK_OVERLAP_RATIO（默认 15%）。"""
    return int(target_chars * CHUNK_OVERLAP_RATIO)


# sentence/section 粒度不需要 overlap（句子原子打包 / 整章一块）
GRANULARITY_CONFIG = {
    "sentence":  {"target_chars": 260, "max_chars": 420, "overlap_chars": 0},
    "paragraph": {"target_chars": 650, "max_chars": 900,
                  "overlap_chars": _overlap_for(650)},  # 650*15% ≈ 97 字符
    "section":   {"target_chars": None, "max_chars": 2600, "overlap_chars": 0},
}
# 多粒度顺序：先生成细粒度，块序号按粒度独立编号
DEFAULT_GRANULARITIES = ["sentence", "paragraph", "section"]
_GRAN_PREFIX = {"sentence": "sen", "paragraph": "par", "section": "sec"}

# 结论/创新点/研究贡献章节关键词（用于分块保护与标题识别）
_INNOVATION_SECTION_KEYWORDS = frozenset({
    "结论", "创新点", "研究贡献", "贡献", "创新", "小结", "总结",
    "conclusion", "discussion", "summary", "contribution", "conclusions",
})


def _is_innovation_section(section: dict) -> bool:
    """判断是否为创新点/结论/研究贡献相关章节（需整段保护，避免截断）。"""
    name = section.get("name", "")
    heading = section.get("heading", "")
    if name in {"conclusion", "discussion", "summary", "other"}:
        return True
    return any(kw in heading for kw in _INNOVATION_SECTION_KEYWORDS)

MIN_CHUNK_CHARS = 40     # 短于该阈值的块直接丢弃（无检索价值）
SECTION_TRUNCATE_MARK = "……【章节过长，已截断】"


# ==========================================================================
# 基础切分工具
# ==========================================================================
def _split_sentences(text: str) -> list:
    """
    按中英文句末标点切分句子（保留标点）。

    细节：小数点（如 3.14）后无空白，不会被切分；换行视为软边界。
    """
    if not text:
        return []
    parts = re.split(r"(?<=[。！？!?])\s*|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _join(a: str, b: str) -> str:
    """拼接两句：英文句间补空格，中文直接连接。"""
    if not a:
        return b
    if not b:
        return a
    if a[-1].isascii() and a[-1].isalnum() and b[0].isascii() and b[0].isalnum():
        return a + " " + b
    return a + b


def _find_sentence_boundary(text: str, lo: int, hi: int) -> int:
    """
    在 text[lo:hi] 中找最后一个句末标点位置，返回切分点（标点后一位）；
    找不到则返回 lo（允许硬切，保证滑窗可推进）。
    """
    best = lo
    for m in re.finditer(r"[。！？!?]", text[lo:hi]):
        best = lo + m.end()
    return best


def _split_sliding(text: str, target: int, max_chars: int, overlap: int) -> list:
    """
    滑窗切分：先取 target 长度，再在 [target, max_chars] 范围内寻找句边界
    对齐切分点；下一次起点回退 overlap 字符形成块间重叠。
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + target, n)
        if end < n:  # 尚未到文本尾部：向右寻找句边界（最多延展到 max_chars）
            end = _find_sentence_boundary(text, end, min(start + max_chars, n))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)  # 防御：保证起点前进
    return chunks


def _pack_sentences(sentences: list, target: int, max_chars: int) -> list:
    """
    句子打包：将句子按顺序装入约 target 字符的块；超长单句先用滑窗拆开。
    """
    chunks, cur = [], ""
    for sent in sentences:
        if len(sent) > max_chars:  # 极长句（如公式推导整段）：滑窗强拆
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.extend(_split_sliding(sent, target, max_chars, 0))
            continue
        if cur and len(cur) + len(sent) + 1 > target:
            chunks.append(cur)
            cur = sent
        else:
            cur = _join(cur, sent)
    if cur:
        chunks.append(cur)
    return chunks


def _split_paragraph(text: str, cfg: dict) -> list:
    """
    段落级切分。

    模式选择（CHUNK_MODE / `python -m src.chunking --mode`）：
    - semantic: 语义分块（TextTiling 风格，在相邻窗口相似度谷底切分），
      让块边界落在话题转换处；失败时自动回退字符滑窗；
    - char: 字符滑窗（默认，历史行为）——优先使用 LlamaIndex
      SentenceSplitter（token 感知、句边界更稳健）；未安装或切分异常时
      回退到内置滑窗。
    """
    if CHUNK_MODE == "semantic":
        pieces = _split_semantic(text, cfg["target_chars"], cfg["max_chars"])
        if pieces:
            return pieces
        print("[chunking] 语义分块未产出结果，本节回退字符滑窗。")
    if _HAS_LLAMA_INDEX:
        try:
            # SentenceSplitter 的 chunk_size 以 token 计，中英文下与字符数
            # 大致同量级，这里直接用字符目标作为近似值（启发式，够用）。
            splitter = SentenceSplitter(
                chunk_size=cfg["target_chars"],
                chunk_overlap=cfg["overlap_chars"],
            )
            pieces = [p.strip() for p in splitter.split_text(text) if p.strip()]
            if pieces:
                return pieces
        except Exception as e:  # LlamaIndex 切分失败不影响主流程
            print(f"[chunking] SentenceSplitter 切分失败，回退内置切分: {e}")
    return _split_sliding(text, cfg["target_chars"], cfg["max_chars"], cfg["overlap_chars"])


# ==========================================================================
# 语义分块（TextTiling 风格，零第三方依赖）
# ==========================================================================
def _merge_token_sets(sets_list: list) -> set:
    """合并多个词集合。"""
    out = set()
    for s in sets_list:
        out |= s
    return out


def _set_cosine(a: set, b: set) -> float:
    """集合余弦（词集合的重叠度度量，稀疏词袋的轻量近似）。"""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / ((len(a) * len(b)) ** 0.5)


def _units_text(units: list, i: int, j: int) -> str:
    """把 units[i:j] 按中英文边界拼接为一个文本块。"""
    out = ""
    for u in units[i:j]:
        out = _join(out, u)
    return out


def _split_semantic(text: str, target: int, max_chars: int) -> list:
    """
    轻量语义分块（TextTiling 风格）：

    1. 句子聚成约 60 字符的"比较单元"；
    2. 相邻窗口（左右各 k 个单元）做词集合余弦相似度——话题转换处出现
       相似度谷底；
    3. 游走累计长度：进入 [0.8*target, max_chars] 区间后，在允许范围内
       选择相似度最低的边界切分——块大小可控且边界落在语义断层上。

    词表过稀/文本过短/任何异常时返回 []，由调用方回退字符滑窗。
    """
    try:
        from .retrieval import tokenize
    except ImportError:  # 以脚本方式直接运行
        from retrieval import tokenize

    try:
        sents = _split_sentences(text)
        if not sents:
            return []
        # 句子 -> 比较单元（约 60 字符，粒度太细则相似度噪声大）
        units, cur = [], ""
        for s in sents:
            if cur and len(cur) + len(s) > 60:
                units.append(cur)
                cur = s
            else:
                cur = _join(cur, s)
        if cur:
            units.append(cur)

        unit_chars = [len(u) for u in units]
        char_at = [0]
        for n in unit_chars:
            char_at.append(char_at[-1] + n)
        total = char_at[-1]

        if total <= max_chars and len(units) <= 2:
            return [text.strip()]
        if len(units) < 3:
            return []  # 单元太少无法找语义边界，交回退路径

        # 相邻窗口相似度: gaps[i] = 单元 i 与 i+1 之间边界的语义连贯度
        k = 2  # 每侧窗口单元数
        unit_tokens = [set(tokenize(u)) for u in units]
        gaps = []
        for i in range(len(units) - 1):
            left = _merge_token_sets(unit_tokens[max(0, i - k + 1): i + 1])
            right = _merge_token_sets(unit_tokens[i + 1: min(len(units), i + 1 + k)])
            gaps.append(_set_cosine(left, right))

        # 游走切分：累计长度达到 0.8*target 后，允许范围内挑相似度最低的边界。
        # 相似度并列时取更晚的边界（<=），使块尽量接近 target、避免碎片小块。
        pieces, start = [], 0
        while start < len(units):
            # 剩余整体已不超过 max_chars：不再切分，整体收尾（避免尾碎块）
            if char_at[-1] - char_at[start] <= max_chars:
                pieces.append(_units_text(units, start, len(units)))
                break
            lo_len = char_at[start] + int(target * 0.8)
            best_j, best_sim = None, None
            j = start + 1
            while j < len(units) and char_at[j] - char_at[start] <= max_chars:
                if char_at[j] >= lo_len and j - 1 < len(gaps):
                    sim = gaps[j - 1]
                    if best_sim is None or sim <= best_sim:
                        best_j, best_sim = j, sim
                j += 1
            if best_j is None:
                # 允许窗口内无可切边界（剩余偏短）：整体收尾
                pieces.append(_units_text(units, start, len(units)))
                break
            pieces.append(_units_text(units, start, best_j))
            start = best_j

        # 大小约束兜底：超长块滑窗强拆，空块丢弃
        result = []
        for p in pieces:
            p = p.strip()
            if not p:
                continue
            if len(p) <= max_chars:
                result.append(p)
            else:
                result.extend(_split_sliding(p, target, max_chars, _overlap_for(target)))
        return result
    except Exception as e:  # 语义分块任何异常都不阻塞主流程
        print(f"[chunking] 语义分块异常，回退字符滑窗: {e}")
        return []


def _split_innovation_paragraph(text: str, max_chars: int) -> list:
    """
    创新点/结论/研究贡献章节专用切分：
    - 文本 <= max_chars 时整段返回，保证创新内容不被截断；
    - 否则优先按段落（双换行）切分，尽量整段保留；
    - 段落仍超长时，按句号/问号/感叹号边界切分，整句保留；
    - 单句仍超长时才滑窗强拆（最后手段）。
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    # 优先按段落切分
    if "\n\n" in text:
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(parts) >= 2:
            merged, cur = [], ""
            for p in parts:
                if len(cur) + len(p) + 2 <= max_chars:
                    cur = cur + "\n\n" + p if cur else p
                else:
                    if cur:
                        merged.append(cur)
                    cur = p
            if cur:
                merged.append(cur)
            if merged:
                return merged

    # 按句末标点切分，尽量整句保留
    sents = _split_sentences(text)
    packed = _pack_sentences(sents, max_chars, max_chars)
    if packed:
        return packed

    # 最后手段：滑窗强拆
    return _split_sliding(text, max_chars, max_chars, 0)


# ==========================================================================
# 页码反查
# ==========================================================================
def _page_for_offset(page_spans: list, offset: int) -> int:
    """
    给定字符偏移在 section 内的位置，反查所在页码。

    page_spans 结构: [{"page": 3, "start": 0, "end": 812}, ...]（按 start 升序）。
    偏移落在某 span 内直接返回；落在空隙时返回前一个 span 的页码。
    """
    if not page_spans:
        return 1
    starts = [s["start"] for s in page_spans]
    idx = bisect_right(starts, offset) - 1
    if idx < 0:
        return page_spans[0]["page"]
    return page_spans[idx]["page"]


# ==========================================================================
# chunk 构造
# ==========================================================================
def chunk_section(section: dict, paper: dict, granularities=None, counters=None) -> list:
    """
    对单个 section 生成多粒度 chunk。

    Args:
        section: 解析 JSON 中的单个章节（含 text 与 page_spans）。
        paper:   整篇论文解析结果（取 paper_id / title）。
        granularities: 需要的粒度列表，默认全部三种。
        counters: 各粒度的序号计数器（由 chunk_paper 持有，保证同一论文
                  内 chunk_id 全局唯一）；为 None 时独立计数（单章调试用）。

    Returns:
        chunk 记录列表（同粒度内按正文顺序编号）。
    """
    granularities = granularities or DEFAULT_GRANULARITIES
    text = section.get("text", "").strip()
    if len(text) < MIN_CHUNK_CHARS:
        return []

    spans = section.get("page_spans", [])
    chunks = []
    counters = counters if counters is not None else {}

    for gran in granularities:
        cfg = GRANULARITY_CONFIG[gran]
        if gran == "sentence":
            pieces = _pack_sentences(_split_sentences(text), cfg["target_chars"], cfg["max_chars"])
        elif gran == "paragraph":
            # 创新点/结论/研究贡献章节使用专用切分，避免在段落中间截断
            if _is_innovation_section(section):
                pieces = _split_innovation_paragraph(text, cfg["max_chars"])
            else:
                pieces = _split_paragraph(text, cfg)
        else:  # section 粒度：整章一块，超长截断（粗粒度供大上下文问答）
            pieces = [text if len(text) <= cfg["max_chars"] else text[:cfg["max_chars"]] + SECTION_TRUNCATE_MARK]

        for piece in pieces:
            piece = piece.strip()
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            counters[gran] = counters.get(gran, 0) + 1
            seq = counters[gran]
            chunks.append({
                # 全局唯一 ID：论文ID-粒度-序号，可直接反查检索证据
                "chunk_id": f"{paper['paper_id']}-{_GRAN_PREFIX[gran]}-{seq:04d}",
                "paper_id": paper["paper_id"],
                "paper_title": paper.get("title", "Unknown Title"),
                # 元数据：原始 PDF 文件名（检索 filter 过滤 / 引用溯源用）
                "source": paper.get("filename", ""),
                "section": section["name"],
                "section_heading": section.get("heading", ""),
                "granularity": gran,
                "text": piece,
                # 元数据：所在页码（由 page_spans 反查）
                "page": _page_for_offset(spans, text.find(piece[:30]) if piece[:30] in text else 0),
                "char_count": len(piece),
                "seq": seq,
            })
    return chunks


def chunk_paper(parsed: dict, granularities=None) -> list:
    """对单篇论文的全部章节做分块（序号计数器跨章节共享，保证ID唯一）。"""
    chunks, counters = [], {}
    for section in parsed.get("sections", []):
        chunks.extend(chunk_section(section, parsed, granularities, counters))
    return chunks


def build_all_chunks(parsed_dir=PARSED_DIR, out_path=CHUNKS_FILE) -> list:
    """
    批量分块：读取 data/parsed/*.json，生成全部 chunk 并写入
    data/chunks/chunks.json（检索索引以该文件为唯一语料来源）。
    """
    ensure_dirs()
    parsed_dir, out_path = Path(parsed_dir), Path(out_path)
    files = sorted(parsed_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"{parsed_dir} 下没有解析结果，请先运行: python -m src.main parse"
        )

    all_chunks = []
    for f in tqdm(files, desc="多粒度分块"):
        parsed = json.loads(f.read_text(encoding="utf-8"))
        all_chunks.extend(chunk_paper(parsed))

    if not all_chunks:
        raise ValueError("分块结果为空：请检查 PDF 是否包含可抽取的文本层（扫描版 PDF 需先 OCR）。")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=1), encoding="utf-8")

    # 打印分块统计，便于核对粒度分布
    by_gran = {}
    for c in all_chunks:
        by_gran[c["granularity"]] = by_gran.get(c["granularity"], 0) + 1
    print(f"[chunking] 共 {len(all_chunks)} 个分块 ({by_gran}) -> {out_path}")
    return all_chunks


if __name__ == "__main__":
    setup_console()
    import argparse
    ap = argparse.ArgumentParser(
        description="多粒度分块（--mode semantic 启用语义分块，默认字符滑窗）")
    ap.add_argument("--mode", choices=["char", "semantic"], default=None,
                    help="分块模式: char=字符滑窗（默认）；semantic=语义分块。"
                         "缺省读取 CHUNK_MODE 环境变量。注意: 切分结果变化后"
                         "需运行 python -m src.main index 重建（仅变更块会重新向量化）")
    args = ap.parse_args()
    if args.mode:
        CHUNK_MODE = args.mode  # 覆盖模块级默认（仅本次运行生效）
    print(f"[chunking] 分块模式: {CHUNK_MODE} | overlap比例: {CHUNK_OVERLAP_RATIO:.0%}")
    build_all_chunks()
