# -*- coding: utf-8 -*-
"""
PDF 结构化解析模块。

功能
----
1. 使用 pdfplumber 逐页抽取文本；
2. 自动识别并去除页眉/页脚噪声（跨页重复行、纯页码行、arXiv 标识等）；
3. 依据标题启发式规则识别论文章节，并归一到规范名称：
   abstract / introduction / related_work / method / experiment /
   results / discussion / conclusion / references / other
   （覆盖需求中的 abstract/intro/method/experiment/conclusion 五大类）；
4. 每个 section 记录页码区间与字符偏移（page_spans），供下游分块时
   反查"该块位于第几页"，支撑引用溯源；
5. 批量处理 papers/ 目录下全部 PDF，输出 JSON 到 data/parsed/。

输出 JSON 结构
--------------
{
  "paper_id":   "a1b2c3d4e5f6",          # 文件名 SHA-1 前 12 位，全局唯一
  "filename":   "demo.pdf",
  "source_file": "papers/demo.pdf",
  "title":      "论文标题（启发式猜测）",
  "num_pages":  12,
  "sections": [
      {
        "name": "method",                 # 规范章节名
        "heading": "3. Proposed Method",  # 原文标题行
        "page_start": 3, "page_end": 5,
        "text": "...",                    # 该章节正文（换行保留）
        "page_spans": [{"page": 3, "start": 0, "end": 812}, ...]
      }, ...
  ],
  "stats": {"num_chars": 45678, "num_sections": 9, "removed_noise_lines": 34}
}

用法
----
    python -m src.pdf_parser            # 批量解析 papers/ 下全部 PDF
    python -m src.pdf_parser path.pdf   # 解析单个 PDF
"""

import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import pdfplumber

# 支持 `python -m src.pdf_parser` 与 `python src/pdf_parser.py` 两种运行方式
try:
    from .config import DATA_PAPERS_DIR, PARSED_DIR, PAPERS_DIR, ensure_dirs, setup_console
except ImportError:  # 以脚本方式直接运行
    from config import DATA_PAPERS_DIR, PARSED_DIR, PAPERS_DIR, ensure_dirs, setup_console

try:
    from tqdm import tqdm
except ImportError:  # tqdm 缺失时退化为普通循环
    def tqdm(x, **kwargs):
        return x


# ==========================================================================
# 章节识别规则
# ==========================================================================
# 标题行允许携带的编号前缀，例如 "3." / "3.2" / "IV." / "三、" / "A)"
_HEADING_PREFIX = (
    r"^(?:(?:\d{1,2}(?:\.\d{1,2})*"       # 1 / 3.2 / 2.1.3
    r"|[IVXLCivxlc]{1,5}"                 # 罗马数字 I. / IV.
    r"|[一二三四五六七八九十]{1,3}"        # 中文序号 三、
    r"|[A-Za-z]"                          # 子附录编号 A)
    r")[\.．、·)）]?\s*)"
)

# 规范章节名 -> 标题正则（在去掉编号前缀、小写化后匹配）
SECTION_RULES = [
    ("abstract",     r"(?:abstract|summary)\b|^摘\s*要"),
    ("introduction", r"(?:introduction|background)\b|^引\s*言|^绪\s*论|^研究背景"),
    ("related_work", r"(?:related works?|literature review)\b|^相关工作|^文献综述"),
    ("method",       r"(?:methods?|methodology|approach|proposed\s+(?:method|approach|model|framework)"
                     r"|model(?:\s+architecture)?|materials\s+and\s+methods|preliminar(?:y|ies))\b|^方法|^模型|^本文方法"),
    ("experiment",   r"(?:experiments?(?:\s+and\s+.*)?|experimental(?:\s+\w+)*|evaluation|"
                     r"empirical\s+(?:study|evaluation)|datasets?)\b|^实验"),
    ("results",      r"(?:results?(?:\s+and\s+.*)?)\b|^结果"),
    ("discussion",   r"(?:discussion)\b|^讨论|^分析与讨论"),
    ("conclusion",   r"(?:conclusions?|concluding\s+remarks|summary\s+and\s+outlook|future\s+work)\b|^结\s*论|^总结|^结论与展望"),
    ("references",   r"(?:references|bibliography)\b|^参\s*考\s*文\s*献"),
]
_SECTION_RULES_COMPILED = [(name, re.compile(_HEADING_PREFIX + pattern)) for name, pattern in SECTION_RULES]

# 针对被 PDF 抽取破坏了间距的标题（如 "A B S T R A C T"）的兜底映射
_COMPACT_HEADINGS = {
    "abstract": "abstract", "summary": "abstract", "摘要": "abstract",
    "introduction": "introduction", "background": "introduction", "引言": "introduction",
    "绪论": "introduction", "相关工作": "related_work", "文献综述": "related_work",
    "method": "method", "methods": "method", "methodology": "method", "approach": "method",
    "proposedmethod": "method", "方法": "method", "模型": "method",
    "experiment": "experiment", "experiments": "experiment", "evaluation": "experiment",
    "实验": "experiment",
    "results": "results", "结果": "results",
    "discussion": "discussion", "讨论": "discussion",
    "conclusion": "conclusion", "conclusions": "conclusion", "结论": "conclusion", "总结": "conclusion",
    "references": "references", "bibliography": "references", "参考文献": "references",
}

# 页面噪声行特征：纯页码（含罗马数字页码）、arXiv 标识、Page x 等
_NOISE_LINE_RE = re.compile(
    r"^(?:arxiv:\d{4}\.\d{4,5}(?:v\d+)?"
    r"|\d{1,4}"
    r"|[ivxlcdm]{1,7}"
    r"|page\s*\d+(?:\s*/\s*\d+)?"
    r"|第\s*\d+\s*页"
    r")$"
)

_HEADING_MAX_LEN = 80      # 标题行长度上限（字符）
_HEADING_MAX_WORDS = 12    # 标题行单词数上限（防止误伤正文长句）


def _norm_line(line: str) -> str:
    """紧凑归一化：去所有空白并小写，用于页眉页脚/标题的鲁棒匹配。"""
    return re.sub(r"\s+", "", line).lower()


def match_section_heading(line: str):
    """
    判断一行文本是否为章节标题。

    启发式规则（按序判断）：
    1. 行长 <= 80 字符、单词数 <= 12，且不含 @ / http（排除邮箱、链接行）；
    2. 去除尾部句点后，先查"压缩标题"表（处理 A B S T R A C T 这类被拆散的标题）；
    3. 再用"编号前缀 + 关键词"正则匹配，命中即返回规范章节名。

    Returns:
        规范章节名（如 "method"）；不是标题则返回 None。
    """
    text = line.strip().rstrip(".。")
    if not text or len(text) > _HEADING_MAX_LEN:
        return None
    if "@" in text or "http" in text.lower():
        return None
    if len(text.split()) > _HEADING_MAX_WORDS:
        return None

    compact = _norm_line(text)
    if compact in _COMPACT_HEADINGS:
        return _COMPACT_HEADINGS[compact]

    low = text.lower()
    for name, pattern in _SECTION_RULES_COMPILED:
        if pattern.match(low):
            return name
    return None


# ==========================================================================
# 页眉页脚检测
# ==========================================================================
def detect_headers_footers(pages, top_k: int = 2, bottom_k: int = 2,
                           min_ratio: float = 0.3, min_count: int = 2):
    """
    检测页眉/页脚噪声行。

    原理：页眉页脚通常在大多数页面的顶部/底部重复出现。统计每页顶部 top_k 行
    与底部 bottom_k 行的紧凑形式，出现次数达到 max(min_count, 页数*ratio) 的
    判定为噪声。

    Returns:
        set[str]: 噪声行的紧凑归一化形式（与 _norm_line 输出对齐）。
    """
    n = len(pages)
    counter = Counter()
    for pg in pages:
        lines = [ln for ln in pg["lines"] if ln.strip()]
        candidates = lines[:top_k] + (lines[-bottom_k:] if len(lines) > bottom_k else [])
        for line in candidates:
            norm = _norm_line(line)
            if not norm or norm.isdigit():
                continue
            counter[norm] += 1
    threshold = max(min_count, math.ceil(min_ratio * n))
    return {norm for norm, cnt in counter.items() if cnt >= threshold}


def _is_noise_line(line: str) -> bool:
    """单行噪声判定：纯页码 / arXiv 标识 / Page x 等（无需跨页统计）。"""
    norm = _norm_line(line)
    return bool(norm) and bool(_NOISE_LINE_RE.match(norm))


# ==========================================================================
# PDF 读取与解析
# ==========================================================================
def read_pdf_pages(pdf_path) -> list:
    """
    逐页抽取 PDF 文本行。

    Returns:
        [{"page_num": int, "lines": [str, ...]}, ...]
    """
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            # extract_text 对扫描版/无文本层 PDF 可能返回 None
            text = page.extract_text() or ""
            lines = [ln.rstrip() for ln in text.splitlines()]
            pages.append({"page_num": idx, "lines": lines})
    return pages


def _new_section(name: str, heading: str, page_num: int) -> dict:
    """构造一个进行中的 section 骨架。"""
    return {
        "name": name,
        "heading": heading,
        "page_start": page_num,
        "page_end": page_num,
        "pieces": [],  # [(page_num, line), ...]，最终拼装为 text + page_spans
    }


def _finalize_section(sec: dict) -> dict:
    """
    将进行中的 section 组装为最终形态：
    - text: 各行以换行拼接；
    - page_spans: 记录每页文本在该 section 字符串中的 [start, end) 区间，
      供分块阶段按字符偏移反查页码（引用溯源的关键）。
    """
    text_parts, spans, cursor = [], [], 0
    for page_num, line in sec["pieces"]:
        piece = line if not text_parts else "\n" + line
        start, end = cursor, cursor + len(piece)
        if spans and spans[-1]["page"] == page_num:
            spans[-1]["end"] = end  # 同页连续文本合并区间
        else:
            spans.append({"page": page_num, "start": start, "end": end})
        text_parts.append(piece)
        cursor = end
    return {
        "name": sec["name"],
        "heading": sec["heading"],
        "page_start": sec["page_start"],
        "page_end": max(p for _, p in sec["pieces"]) if sec["pieces"] else sec["page_start"],
        "text": "".join(text_parts).strip(),
        "page_spans": spans,
    }


# 学位论文封面元信息行的字段前缀（紧凑归一化、截取冒号前部分后匹配）
_COVER_META_PREFIXES = (
    "学校代码", "学号", "中图分类法", "分类号", "密级", "udc",
    "学位申请人", "指导教师", "所属专业", "所在学院", "所在单位",
    "培养单位", "提交日期", "答辩日期", "论文日期", "研究方向",
    "作者姓名", "学生姓名", "学科专业", "一级学科", "二级学科",
)
# 封面上的文档类型说明行（如"专业学位硕士学位论文"，本身不是标题）
_COVER_DOCTYPE_RE = re.compile(
    r"^(?:(?:专业|科学|工程)?学位)?(?:硕士|博士|学士)(?:研究生)?(?:学位)?论文$"
    r"|^(?:硕士|博士|学士)研究生$|^thesis$|^dissertation$",
    re.IGNORECASE,
)


def _is_cover_meta(line: str) -> bool:
    """
    判断是否为学位论文封面的元信息行（学号/导师/学院等）或文档类型行。
    元信息行必须含冒号（"学校代码： 10255"），避免误伤以"专业"等开头
    的正常标题；文档类型行（"XX学位论文"）无冒号，单独用正则识别。
    """
    compact = _norm_line(line)
    if _COVER_DOCTYPE_RE.match(compact):
        return True
    if "：" not in compact and ":" not in compact:
        return False
    key = re.split(r"[：:]", compact, 1)[0]
    return any(key.startswith(p) for p in _COVER_META_PREFIXES)


# DOI 行（期刊站点元信息，不是标题）
_DOI_LINE_RE = re.compile(r"^doi[:：]\s*\S+", re.IGNORECASE)
# 报纸版头（如"中国新闻出版广电报/2025年/4月/3日/第005版"）
_MASTHEAD_RE = re.compile(r"^.*/.*\d{4}\s*年.*/")
# 逐字加空格标题的压缩判定：单字符 token 占比阈值
_SPACED_SINGLE_RATIO = 0.5


def _compact_spaced(line: str) -> str:
    """
    压缩逐字加空格的标题："电 影 中 的 传 统 文 化 I P 开 发 研 究" ->
    "电影中的传统文化IP开发研究"。

    仅当单字符 token 占比 >= 50% 时启用，避免误伤正常中英文混排标题的
    词间空格（如 "基于 Transformer-LSTM 的 多变量..." 不受影响）。
    """
    tokens = [t for t in line.split(" ") if t]
    if len(tokens) < 4:
        return line
    singles = sum(1 for t in tokens if len(t) == 1)
    if singles / len(tokens) < _SPACED_SINGLE_RATIO:
        return line
    out = ""
    for t in tokens:
        if len(t) == 1:
            out += t
        elif out and out[-1].isascii() and t[0].isascii():
            out += " " + t  # 连续英文单词之间保留空格
        else:
            out += t
    return out


def _is_cjk(ch: str) -> bool:
    """判断字符是否为中日韩统一表意文字（用于中文标题拼接时免加空格）。"""
    return "\u4e00" <= ch <= "\u9fff"


def _is_journal_meta(line: str) -> bool:
    """
    判定是否为期刊页眉/卷期信息（非论文标题）。
    过滤：年份、卷号(Vol/Volume)、期号(No/Issue)、页码(Page)、
    期刊英文名/缩写、ISSN、CN 刊号、DOI 链接、文章编号等。
    """
    compact = _norm_line(line)
    if not compact:
        return False
    # 纯数字/纯字母短串（如 "2025"、"Vol.12"、"No.3"、"123-456"）
    if re.fullmatch(r"[\d\s\-.]+", compact):
        return True
    # 期刊名特征（含 Vol/No/Page/ISSN/CN/Digital Object Identifier 等）
    journal_kw = re.compile(
        r"vol(?:ume)?\.?\s*\d|no\.?\s*\d|page\s*\d|issn\s*\d|"
        r"cn\s*\d+|digital\s+object\s+identifier|doi\s*[:：]|"
        r"^\d{4}\s*年|第\s*\d+\s*卷|第\s*\d+\s*期|\(\d{4}\)$|"
        r"文章编号|文章标识码|中图分类号",
        re.IGNORECASE,
    )
    if journal_kw.search(compact):
        return True
    # 过短且含点号分隔（如 "J. Am. Chem. Soc." 期刊名）
    tokens = compact.split()
    if len(tokens) >= 2 and all(t[0].isupper() for t in tokens if t.isalpha()):
        return True
    return False


def _guess_title(pages, pdf_path=None) -> str:
    """
    启发式猜测论文标题（v3，支持 PDF 元数据优先）：
    ① 优先读取 PDF 元数据 Title 字段；
    ② 若元数据无效，从首页正文识别最大字号行作为标题；
    ③ 过滤期刊卷期/页码/年份/英文刊名等页眉噪声；
    ④ 若仍失败，兜底使用 PDF 文件名。
    """
    # ---- ① PDF 元数据 Title ----
    meta_title = ""
    if pdf_path is not None:
        try:
            with pdfplumber.open(str(pdf_path)) as _pdf:
                raw_meta = _pdf.metadata or {}
                meta_title = (raw_meta.get("Title") or raw_meta.get("title") or "").strip()
        except Exception:
            meta_title = ""
    if meta_title and len(meta_title) >= 4 and not _is_journal_meta(meta_title):
        print(f"[pdf_parser] 元数据标题: {meta_title}")
        return meta_title

    # ---- ② 首页正文识别 ----
    first_page = pages[0] if pages else {"lines": []}
    noisy = detect_headers_footers(pages)
    cands = []
    for raw in first_page["lines"]:
        line = raw.strip()
        if (not line or _is_noise_line(line) or _norm_line(line) in noisy
                or _is_cover_meta(line)
                or _DOI_LINE_RE.match(line)
                or _MASTHEAD_RE.search(line)
                or "本报记者" in line
                or _is_journal_meta(line)):
            continue
        cands.append(_compact_spaced(line))
        if len(cands) >= 10:
            break

    for i, line in enumerate(cands):
        core = line.rstrip("：:")
        if len(core) < 8 or "@" in core or core.isdigit():
            continue
        nxt = cands[i + 1] if i + 1 < len(cands) else ""
        if line.endswith(("：", ":")) and nxt:
            print(f"[pdf_parser] 正文识别标题(冒号副标题): {core}：{nxt}")
            return f"{core}：{nxt}".strip()
        if (len(core) < 60 and nxt
                and len(nxt) >= 8
                and "," not in nxt and "，" not in nxt
                and line[-1] not in ".。!？?，,"):
            sep = "" if (nxt.startswith("——")
                         or (_is_cjk(line[-1]) and _is_cjk(nxt[0]))) else " "
            print(f"[pdf_parser] 正文识别标题(换行拼接): {core}{sep}{nxt}")
            return f"{line}{sep}{nxt}".strip()
        print(f"[pdf_parser] 正文识别标题(单行): {core}")
        return core

    # ---- ③ 兜底：文件名 ----
    fallback = Path(pdf_path).stem if pdf_path else "Unknown Title"
    print(f"[pdf_parser] 标题识别失败，兜底使用文件名: {fallback}")
    return fallback


def parse_pdf(pdf_path) -> dict:
    """
    解析单个 PDF -> 结构化 dict（章节切分 + 去噪 + 页码追踪）。

    解析主循环：按页 -> 按行流式扫描；命中标题行则开启新 section，
    否则将行写入当前 section；首次标题之前的行归入 "other"（标题/作者区）。
    进入 references 章节后不再做标题识别（参考文献条目常以 Introduction
    等词开头，会造成误切）。
    """
    pdf_path = Path(pdf_path)
    pages = read_pdf_pages(pdf_path)
    if not pages:
        raise ValueError(f"PDF 无可抽取内容: {pdf_path}")

    noisy = detect_headers_footers(pages)
    sections, cur, removed = [], None, 0

    for pg in pages:
        page_num = pg["page_num"]
        for raw in pg["lines"]:
            line = raw.strip()
            if not line:
                continue
            # ---- 噪声过滤：页眉/页脚/页码/arXiv 标识 ----
            if _is_noise_line(line) or _norm_line(line) in noisy:
                removed += 1
                continue
            # ---- 标题识别（references 章节内跳过，防止参考文献被误切） ----
            heading_name = None
            if cur is None or cur["name"] != "references":
                heading_name = match_section_heading(line)
            if heading_name:
                cur = _new_section(heading_name, line, page_num)
                sections.append(cur)
                continue
            # ---- 正文行写入当前 section ----
            if cur is None:  # 首个标题之前的行（标题/作者/单位区）
                cur = _new_section("other", "(Front Matter)", page_num)
                sections.append(cur)
            cur["pieces"].append((page_num, line))

    # 相邻同名 section 合并（如 "3.1 Model" 与 "3.2 Training" 都归 method）
    merged = []
    for sec in sections:
        if merged and merged[-1]["name"] == sec["name"]:
            merged[-1]["pieces"].extend(sec["pieces"])
            # 标题行合并展示（超长则截断，避免标题无限膨胀）
            if len(merged[-1]["heading"]) < 60:
                merged[-1]["heading"] += f" / {sec['heading']}"
        else:
            merged.append(sec)
    final_sections = [_finalize_section(s) for s in merged if s["pieces"]]

    total_chars = sum(len(s["text"]) for s in final_sections)
    filename = pdf_path.name
    paper_id = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:12]

    return {
        "paper_id": paper_id,
        "filename": filename,
        "source_file": str(pdf_path),
        "title": _guess_title(pages, pdf_path=pdf_path),
        "num_pages": len(pages),
        "sections": final_sections,
        "stats": {
            "num_chars": total_chars,
            "num_sections": len(final_sections),
            "removed_noise_lines": removed,
        },
    }


def save_paper_json(parsed: dict, out_dir=PARSED_DIR) -> Path:
    """将解析结果写入 data/parsed/<paper_id>.json（UTF-8、保留中文原文）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{parsed['paper_id']}.json"
    out_path.write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


# ==========================================================================
# 批量处理
# ==========================================================================
def batch_parse(papers_dir=PAPERS_DIR, out_dir=PARSED_DIR,
                overwrite: bool = False) -> list:
    """
    批量解析 papers/ 目录下全部 PDF。

    Args:
        overwrite: False（默认）为增量模式——已存在解析结果
                   （data/parsed/<paper_id>.json）的 PDF 跳过，只解析新增
                   的 PDF；True 强制全部重新解析（论文更换/解析规则升级
                   后使用）。paper_id 由文件名哈希决定，与内容无关。

    Returns:
        本次实际解析的结果 dict 列表（单篇失败不影响其余论文）。
    """
    ensure_dirs()
    papers_dir, out_dir = Path(papers_dir), Path(out_dir)
    pdf_files = sorted(papers_dir.glob("*.pdf")) + sorted(papers_dir.glob("*.PDF"))
    # 默认扫描 papers/ 时，额外并入网页上传目录 data/papers/（二者并列生效）
    if papers_dir == PAPERS_DIR and out_dir == PARSED_DIR:
        pdf_files += sorted(DATA_PAPERS_DIR.glob("*.pdf")) + sorted(DATA_PAPERS_DIR.glob("*.PDF"))
    # 默认扫描 papers/ 时，额外并入网页上传目录 data/papers/（二者并列生效）
    if papers_dir == PAPERS_DIR and out_dir == PARSED_DIR:
        pdf_files += sorted(DATA_PAPERS_DIR.glob("*.pdf")) + sorted(DATA_PAPERS_DIR.glob("*.PDF"))
    # 去重（Windows 文件系统大小写不敏感时可能重复）
    seen, unique = set(), []
    for p in pdf_files:
        key = p.name.lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    pdf_files = unique

    if not pdf_files:
        print(f"[pdf_parser] {papers_dir} 下未发现 PDF 文件，请先将要研读的论文放入该目录。")
        return []

    results, failed, skipped = [], [], []
    for pdf_path in tqdm(pdf_files, desc="解析PDF"):
        # 增量解析：paper_id = 文件名SHA-1前12位，对应 JSON 已存在则跳过
        paper_id = hashlib.sha1(pdf_path.name.encode("utf-8")).hexdigest()[:12]
        out_json = Path(out_dir) / f"{paper_id}.json"
        if out_json.exists() and not overwrite:
            skipped.append(pdf_path.name)
            continue
        try:
            parsed = parse_pdf(pdf_path)
            save_paper_json(parsed, out_dir)
            results.append(parsed)
            sec_names = ",".join(s["name"] for s in parsed["sections"])
            print(f"  [OK] {pdf_path.name}: {parsed['num_pages']}页, "
                  f"{parsed['stats']['num_sections']}个章节 [{sec_names}]")
        except Exception as e:  # 单篇失败不中断批量任务
            failed.append(pdf_path.name)
            print(f"  [FAIL] {pdf_path.name}: {e}")

    if skipped:
        print(f"[pdf_parser] 跳过已解析 {len(skipped)} 篇: {', '.join(skipped)}"
              f"（强制重解析请加 --force）")
    print(f"[pdf_parser] 完成: 新解析 {len(results)} 篇, 失败 {len(failed)} 篇; "
          f"JSON 输出目录: {out_dir}")
    return results


if __name__ == "__main__":
    setup_console()
    # 命令行: 可选传入单个 PDF 路径；缺省批量解析 papers/
    if len(sys.argv) > 1:
        parsed = parse_pdf(sys.argv[1])
        out = save_paper_json(parsed)
        print(f"已保存: {out}")
    else:
        batch_parse()
