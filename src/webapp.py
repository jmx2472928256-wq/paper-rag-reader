# -*- coding: utf-8 -*-
"""
Web 前后端模块：FastAPI 后端 API + Gradio 前端页面（同进程、同端口）。

架构约定
--------
Web 只是调用现有 CLI 底层函数的薄封装，不重复实现业务逻辑：
- 服务层（本模块顶部 list_papers / get_summary / generate_summary_for /
  ask_question）复用 config / summarizer / rag_pipeline 的既有函数，
  FastAPI 路由与 Gradio 事件处理器都只调用这一层；
- 检索、摘要、问答、溯源、告警等核心逻辑全部来自 src 包原模块，
  原有 CLI（python -m src.main ...）行为完全不变。

界面
----
1. 文献库：papers/ 目录 + literature_lib 元数据（摘要是否存在）；
2. 单篇查看：选择文献即渲染其 summary Markdown；
3. 【生成摘要】按钮：调用底层 summarizer.generate_summary，完成后刷新展示；
4. 问答：选择单篇（或全部文献），调用 rag.ask，展示回答/溯源/告警，
   并可展开检索片段明细（每块检索分/重排分）；
5. 检索：直接查看检索结果与相似度分数（不调用大模型）；
6. 对比：多篇文献摘要横向对比（缺失摘要自动补生成）。

启动方式
--------
    python web.py        # 自动打开浏览器访问 http://127.0.0.1:7860/
    FastAPI 接口文档: http://127.0.0.1:7860/docs
"""

import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

try:
    from .config import (
        ANSWERS_DIR, COMPARE_DIR, DATA_PAPERS_DIR, INGEST_LOG, PARSED_DIR,
        PAPERS_DIR, SUMMARY_DIR, ensure_dirs, setup_console,
    )
except ImportError:  # 以脚本方式直接运行
    from config import (
        ANSWERS_DIR, COMPARE_DIR, DATA_PAPERS_DIR, INGEST_LOG, PARSED_DIR,
        PAPERS_DIR, SUMMARY_DIR, ensure_dirs, setup_console,
    )

from src import chunking, paper_analysis, pdf_parser, rag_pipeline, retrieval, summarizer

# 问答下拉框中代表"全部文献"的选项值（不作为文件名使用）
ALL_PAPERS = "📚 全部文献（均衡召回）"


# ==========================================================================
# 共享服务层：FastAPI 与 Gradio 都只调用这些函数（内部全部复用 src 原函数）
# ==========================================================================
_rag = None            # RAGPipeline 单例（索引只加载/构建一次）
_rag_lock = threading.Lock()


def get_rag() -> "rag_pipeline.RAGPipeline":
    """懒加载共享的 RAGPipeline（线程安全）。首次调用可能需要构建索引。"""
    global _rag
    with _rag_lock:
        if _rag is None:
            _rag = rag_pipeline.RAGPipeline()
    return _rag


def list_papers(only_indexed: bool = False) -> list:
    """
    文献列表：扫描 papers/ 与 data/papers/（网页上传目录）两个目录，
    合并解析元数据（data/parsed，只读复用 pdf_parser 的产物）与
    literature_lib 摘要状态。

    Args:
        only_indexed: True 时仅返回已在向量索引中的文献（通过检索器
                      chunks 校验，用于上传失败回滚后不显示半成品）。
    """
    ensure_dirs()
    seen, files = set(), []
    for p in (sorted(PAPERS_DIR.glob("*.pdf")) + sorted(PAPERS_DIR.glob("*.PDF"))
              + sorted(DATA_PAPERS_DIR.glob("*.pdf")) + sorted(DATA_PAPERS_DIR.glob("*.PDF"))):
        if p.name.lower() not in seen:  # Windows 大小写不敏感去重
            seen.add(p.name.lower())
            files.append(p)

    # 解析元数据按"原始文件名"索引（解析 JSON 自带 filename 字段）
    parsed_meta = {}
    for j in PARSED_DIR.glob("*.json"):
        try:
            d = json.loads(j.read_text(encoding="utf-8"))
            if d.get("filename"):
                parsed_meta[d["filename"]] = d
        except Exception:
            continue  # 单个损坏文件不影响列表

    # 向量索引中的 source 集合（用于 only_indexed 过滤）
    indexed_sources = set()
    if only_indexed:
        try:
            indexed_sources = set(get_rag().retriever.distinct_sources())
        except Exception:
            pass

    papers = []
    for p in files:
        if only_indexed and p.name not in indexed_sources:
            continue
        meta = parsed_meta.get(p.name, {})
        summary_path = SUMMARY_DIR / f"{p.stem}.md"
        papers.append({
            "filename": p.name,
            "title": meta.get("title") or "(未解析)",
            "num_pages": meta.get("num_pages", 0),
            "parsed": bool(meta),
            "has_summary": summary_path.exists(),
            "summary_path": str(summary_path) if summary_path.exists() else "",
        })
    return papers


def get_summary(name: str) -> dict:
    """读取单篇文献的摘要 Markdown（literature_lib/summary/<主文件名>.md）。"""
    path = SUMMARY_DIR / f"{Path(name).stem}.md"
    exists = path.exists()
    return {
        "exists": exists,
        "path": str(path),
        "content": path.read_text(encoding="utf-8") if exists else "",
    }


def delete_paper(source: str) -> dict:
    """
    删除单篇文献（从知识库彻底移除）。

    删除范围：
    1. data/papers/ 下的 PDF
    2. data/parsed/ 下的解析 JSON（按 paper_id / filename 匹配）
    3. literature_lib/summary/ 下的摘要
    4. 向量索引中的该文献全部分块（重建 chunks.json + FAISS 索引）

    Args:
        source: PDF 文件名（如 paper1.pdf 或中文名.pdf）。

    Returns:
        {"deleted": bool, "detail": str, "table_rows": ..., "dropdown_choices": ...}
    """
    global _rag
    print(f"[webapp] 删除文献: {source}")
    # 先确认该文献是否在索引中
    rag = get_rag()
    indexed = sorted({c.get("source") for c in rag.retriever.chunks if c.get("source")})
    if source not in indexed:
        return {"deleted": False, "detail": f"该文献不在索引中: {source}",
                "table_rows": _table_rows(list_papers(only_indexed=True)),
                "dropdown_choices": [p["filename"] for p in list_papers(only_indexed=True)]}

    # 1. 删除磁盘文件
    removed_files = []
    for p in [DATA_PAPERS_DIR / source, PAPERS_DIR / source]:
        if p.exists():
            p.unlink()
            removed_files.append(str(p))
    # 删除解析 JSON（按 filename 匹配）
    for j in PARSED_DIR.glob("*.json"):
        try:
            if json.loads(j.read_text(encoding="utf-8")).get("filename") == source:
                j.unlink()
                removed_files.append(str(j))
        except Exception:
            pass
    # 删除摘要
    summary_path = SUMMARY_DIR / f"{Path(source).stem}.md"
    if summary_path.exists():
        summary_path.unlink()
        removed_files.append(str(summary_path))

    # 2. 重建索引（排除已删除文献）
    try:
        retriever = retrieval.build_index()
        with _rag_lock:
            _rag = rag_pipeline.RAGPipeline(retriever=retriever)
        print(f"[webapp] 删除成功并重建索引: {source}")
    except Exception as e:
        print(f"[webapp] 删除成功但索引重建失败: {e}")
        return {"deleted": False, "detail": f"文件已删除，但索引重建失败: {e}",
                "table_rows": _table_rows(list_papers(only_indexed=True)),
                "dropdown_choices": [p["filename"] for p in list_papers(only_indexed=True)]}

    papers = list_papers(only_indexed=True)
    return {"deleted": True, "detail": f"已删除 {source}（含 {len(removed_files)} 个文件）并重建索引",
            "table_rows": _table_rows(papers),
            "dropdown_choices": [p["filename"] for p in papers]}


def generate_summary_for(source: str) -> dict:
    """为指定 PDF 生成结构化摘要（复用 summarizer.generate_summary）。"""
    # 前置校验：只允许索引中真实存在的文献生成摘要
    rag = get_rag()
    indexed = sorted({c.get("source") for c in rag.retriever.chunks if c.get("source")})
    if source not in indexed:
        raise ValueError(f"{source} 不在当前索引中，无法生成摘要。"
                         f"请先上传该文献并确保入库成功。可用文件名: {indexed}")
    print(f"[webapp] 生成摘要: {source}（调用GLM-5.3，约需1分钟）")
    return summarizer.generate_summary(source, rag)


def ask_question(question: str, source: str = None, top_k: int = None,
                 rerank: bool = None) -> dict:
    """
    问答（复用 rag_pipeline.ask）。

    Args:
        source: 指定单篇 PDF 文件名则仅针对该篇解答；None/空 表示全部
                文献（走均衡召回，逐篇解答）。
        rerank: Reranker 重排序开关；None 跟随 RERANK_ENABLED 配置。
    """
    rag = get_rag()
    metadata_filter = {"source": source} if source else None
    print(f"[webapp] 问答: source={source or '全部文献'} top_k={top_k} rerank={rerank}")
    return rag.ask(question, top_k=top_k, metadata_filter=metadata_filter,
                   rerank=rerank)


def run_retrieve(query: str, source: str = None, top_k: int = None,
                 rerank: bool = None) -> dict:
    """
    独立检索（不调用大模型）——复用 rag_pipeline.retrieve 的检索/重排
    策略，用于排查召回质量。
    """
    rag = get_rag()
    metadata_filter = {"source": source} if source else None
    print(f"[webapp] 检索调试: source={source or '全部文献'} top_k={top_k} rerank={rerank}")
    hits, info = rag.retrieve(query, top_k=top_k, metadata_filter=metadata_filter,
                              rerank=rerank)
    return {"hits": hits, "info": info}


def run_compare(sources: list, refresh: bool = False) -> dict:
    """多篇横向对比（复用 summarizer.compare_summaries，缺失摘要自动补生成）。"""
    print(f"[webapp] 对比: {sources} refresh={refresh}")
    return summarizer.compare_summaries(sources, get_rag(), refresh=refresh)


def run_compare_viewpoints(question: str, sources: list,
                           top_k_per_paper: int = None,
                           use_saved_summaries: bool = False) -> dict:
    """
    问题驱动的多篇对比（「对比」标签页后端）：
    逐篇限定来源检索 -> 单篇观点总结 -> 汇总横向对比
    （复用 summarizer.compare_viewpoints，底层为现有 FAISS 检索 + GLM-5.3）。
    """
    print(f"[webapp] 观点对比: {sources} | 问题: {question[:30]}")
    return summarizer.compare_viewpoints(question, sources, get_rag(),
                                         top_k_per_paper=top_k_per_paper,
                                         use_saved_summaries=use_saved_summaries)


# ==========================================================================
# 网页上传闭环（情况B 从零新增）：保存 -> 解析 -> 分块 -> 增量追加索引
# ==========================================================================
def _ingest_log(record: dict) -> None:
    """上传/入库日志（logs/ingest.log）：时间戳 + 逐文件结果明细。"""
    ensure_dirs()
    from datetime import datetime as _dt
    lines = [f"[{_dt.now().isoformat(timespec='seconds')}] " + json.dumps(
        record, ensure_ascii=False, default=str)]
    with open(INGEST_LOG, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def ingest_uploaded_files(filepaths: list) -> dict:
    """
    上传入库主流程（Gradio 处理器与 API 共用）。

    对每个上传文件依次执行：扩展名校验 -> PDF 可读性校验 -> 同名查重 ->
    保存到 data/papers/ -> 复用 pdf_parser 解析（元数据: 文件名/页码）->
    chunking 分块 -> retrieval.append_to_index 增量追加 FAISS（不删旧索引）。

    Returns:
        {"results": [{"file", "status", "detail"}...], "n_ok", "n_skip",
         "n_fail", "n_added_chunks", "table_rows", "dropdown_choices"}
        status 取值: ok(入库成功) / duplicate(同名跳过) / invalid(非PDF)
        / broken(损坏PDF) / failed(解析/索引失败)
        table_rows 仅包含已在向量索引中的文献（失败/回滚的不展示）。
    """
    global _rag
    ensure_dirs()
    results, new_parsed = [], []

    for fp in filepaths:
        fp = Path(fp)
        base = {"file": fp.name}
        # ---- 步骤1: 扩展名校验 ----
        if fp.suffix.lower() != ".pdf":
            results.append({**base, "status": "invalid",
                            "detail": "不是PDF文件（仅支持 .pdf）"})
            _ingest_log({**base, "step": "ext_check", "status": "invalid"})
            print(f"[upload] ❌ {fp.name}: 非PDF文件")
            continue
        # ---- 步骤2: PDF 可读性 ----
        try:
            import pdfplumber
            with pdfplumber.open(str(fp)) as _pdf:
                _ = len(_pdf.pages)
            print(f"[upload] 📖 {fp.name}: PDF可读性校验通过")
        except Exception as e:
            results.append({**base, "status": "broken", "detail": f"PDF无法读取: {e}"})
            _ingest_log({**base, "step": "pdf_read", "status": "broken", "error": str(e)})
            print(f"[upload] ❌ {fp.name}: PDF损坏 -> {e}")
            continue
        # ---- 步骤3: 同名查重 ----
        dest = DATA_PAPERS_DIR / fp.name
        if dest.exists() or (PAPERS_DIR / fp.name).exists():
            results.append({**base, "status": "duplicate",
                            "detail": "同名文件已存在（papers/ 或 data/papers/），已跳过"})
            _ingest_log({**base, "step": "dedup", "status": "duplicate"})
            print(f"[upload] ⏭️ {fp.name}: 同名已存在，跳过")
            continue
        # ---- 步骤4: 保存到 data/papers/ ----
        dest.write_bytes(fp.read_bytes())
        print(f"[upload] 💾 {fp.name}: 已保存到 {dest}")
        # ---- 步骤5: PDF文本提取（pdf_parser）----
        try:
            parsed = pdf_parser.parse_pdf(dest)
            pdf_parser.save_paper_json(parsed)
            print(f"[upload] ✅ {fp.name}: 文本提取成功 ({parsed['num_pages']}页, {len(parsed['sections'])}章节)")
        except Exception as e:
            results.append({**base, "status": "failed", "detail": f"PDF解析失败: {e}"})
            _ingest_log({**base, "step": "parse", "status": "failed", "error": str(e)})
            print(f"[upload] ❌ {fp.name}: 解析失败 -> {e}")
            dest.unlink(missing_ok=True)
            continue
        # ---- 步骤6: 文本切片（chunking）----
        try:
            chunks = chunking.chunk_paper(parsed)
            if not chunks:
                raise RuntimeError("解析成功但未产出任何分块（可能无文本层，需先OCR）")
            print(f"[upload] ✅ {fp.name}: 切片成功 ({len(chunks)}个分块)")
        except Exception as e:
            results.append({**base, "status": "failed", "detail": f"切片失败: {e}"})
            _ingest_log({**base, "step": "chunking", "status": "failed", "error": str(e)})
            print(f"[upload] ❌ {fp.name}: 切片失败 -> {e}")
            dest.unlink(missing_ok=True)
            continue
        new_parsed.extend(chunks)
        results.append({**base, "status": "ok",
                        "detail": f"{parsed['num_pages']}页 / {len(chunks)}个分块",
                        "chunks": chunks, "paper_id": parsed.get("paper_id")})
        _ingest_log({**base, "step": "chunking", "status": "ok",
                     "pages": parsed["num_pages"], "chunks": len(chunks),
                     "title": parsed.get("title", "")})

    # ---- 步骤7: 向量化 + 写入FAISS索引（所有ok文件一次性追加）----
    n_added = 0
    if new_parsed:
        try:
            print(f"[upload] 🧠 向量化并写入索引 ({len(new_parsed)}个分块)...")
            retriever, n_added = retrieval.append_to_index(new_parsed)
            with _rag_lock:
                _rag = rag_pipeline.RAGPipeline(retriever=retriever)
            print(f"[upload] ✅ 索引写入成功 (新增{n_added}块)")
        except Exception as e:
            _ingest_log({"stage": "append_to_index", "status": "failed",
                         "error": str(e)})
            print(f"[upload] ❌ 索引写入失败 -> {e}")
            # 回滚：从结果中移除本次上传的文件（不展示半成品到文献列表）
            failed_names = {r["file"] for r in results if r["status"] == "ok"}
            for r in results:
                if r["status"] == "ok":
                    r["status"] = "failed"
                    r["detail"] += f" | 索引写入失败: {e}（文件已保留，可运行 python -m src.main add 重试）"
            # 清理磁盘上的半成品（PDF + 解析JSON）
            for name in failed_names:
                fpath = DATA_PAPERS_DIR / name
                if fpath.exists():
                    fpath.unlink(missing_ok=True)
                stem = Path(name).stem
                for j in PARSED_DIR.glob(f"*{stem}*.json"):
                    j.unlink(missing_ok=True)
            print(f"[upload] 🧹 已回滚清理 {len(failed_names)} 个半成品文件")

    # 只展示已在索引中的文献（失败/回滚的不显示）
    papers = list_papers(only_indexed=True)
    return {
        "results": [{k: v for k, v in r.items() if k != "chunks"} for r in results],
        "n_ok": sum(1 for r in results if r["status"] == "ok"),
        "n_skip": sum(1 for r in results if r["status"] == "duplicate"),
        "n_fail": sum(1 for r in results if r["status"] in ("invalid", "broken", "failed")),
        "n_added_chunks": n_added,
        "table_rows": _table_rows(papers),
        "dropdown_choices": [p["filename"] for p in papers],
    }


# ==========================================================================
# FastAPI 后端（REST 接口与 Gradio 共用上面的服务层）
# ==========================================================================
class SummaryRequest(BaseModel):
    source: str


class AskRequest(BaseModel):
    question: str
    source: str = None
    # 任务⑤：API 层参数边界保护——top_k 限 [1, 50]，非法值由 FastAPI 返回 422
    top_k: int | None = Field(default=None, ge=1, le=50)
    rerank: bool = None


class RetrieveRequest(BaseModel):
    query: str
    source: str = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    rerank: bool = None


class CompareRequest(BaseModel):
    sources: list
    refresh: bool = False


class CompareViewpointsRequest(BaseModel):
    question: str = ""   # 空 = 自动对比模式（五维度全文内容对比）
    sources: list
    # API 层边界保护（每篇检索块数限 [1, 20]）
    top_k_per_paper: int | None = Field(default=None, ge=1, le=20)
    use_saved_summaries: bool = False


def _register_api(app: FastAPI) -> None:
    """注册 REST 接口（薄封装，业务全部在服务层）。"""

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/papers")
    def api_papers():
        return {"papers": list_papers()}

    @app.get("/api/summary/{name}")
    def api_summary(name: str):
        r = get_summary(name)
        if not r["exists"]:
            raise HTTPException(status_code=404, detail=f"摘要不存在: {r['path']}")
        return r

    @app.post("/api/summary/generate")
    def api_generate(req: SummaryRequest):
        try:
            return generate_summary_for(req.source)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/ask")
    def api_ask(req: AskRequest):
        if not req.question or not req.question.strip():
            raise HTTPException(status_code=400, detail="问题不能为空")
        try:
            return ask_question(req.question.strip(), req.source, req.top_k,
                                rerank=req.rerank)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/upload")
    async def api_upload(file: UploadFile):
        """REST 上传端点（单文件；multipart/form-data）。"""
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="仅支持PDF文件")
        ensure_dirs()
        tmp = DATA_PAPERS_DIR / f".upload_{file.filename}"
        try:
            content = await file.read()
            tmp.write_bytes(content)
            res = ingest_uploaded_files([str(tmp)])
            r = res["results"][0]
            if r["status"] != "ok":
                raise HTTPException(status_code=400, detail=r["detail"])
            return {**r, "n_added_chunks": res["n_added_chunks"]}
        finally:
            tmp.unlink(missing_ok=True)  # 临时文件总是清理

    @app.post("/api/retrieve")
    def api_retrieve(req: RetrieveRequest):
        if not req.query or not req.query.strip():
            raise HTTPException(status_code=400, detail="查询词不能为空")
        try:
            return run_retrieve(req.query.strip(), req.source, req.top_k,
                                rerank=req.rerank)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/compare")
    def api_compare(req: CompareRequest):
        if not req.sources or len(req.sources) < 2:
            raise HTTPException(status_code=400, detail="compare 至少需要两篇文献")
        try:
            return run_compare([str(s) for s in req.sources], refresh=req.refresh)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/compare/viewpoints")
    def api_compare_viewpoints(req: CompareViewpointsRequest):
        # 边界判断：文献不足 → 400；问题为空 → 合法（自动对比模式）
        if not req.sources or len(req.sources) < 2:
            raise HTTPException(status_code=400, detail="对比至少需要两篇文献")
        try:
            return run_compare_viewpoints((req.question or "").strip(),
                                          [str(s) for s in req.sources],
                                          top_k_per_paper=req.top_k_per_paper,
                                          use_saved_summaries=req.use_saved_summaries)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))


# ==========================================================================
# Gradio 前端（Blocks 布局，事件处理器只调用服务层）
# ==========================================================================
def _table_rows(papers: list) -> list:
    """把文献列表转为表格行：文件名/标题/页数/已解析/摘要。"""
    return [[
        p["filename"], p["title"], str(p["num_pages"] or "—"),
        "✅" if p["parsed"] else "—",
        "✅ 已生成" if p["has_summary"] else "—",
    ] for p in papers]


def _hit_meta(r: dict) -> dict:
    """
    统一读取命中条目的元数据（兼容两种形状，缺失时给友好占位而非问号）：
    - ask 规整化后的扁平 dict：source/section/page/text 在顶层；
    - 检索器原始命中（retrieve 接口返回）：元数据嵌套在 chunk 字典内。
    """
    c = r.get("chunk") or {}

    def _first(*vals):
        """返回第一个非 None 且非空串的值。"""
        for v in vals:
            if v is not None and v != "":
                return v
        return None

    return {
        "rank": _first(r.get("rank"), c.get("rank")),
        "score": _first(r.get("score"), c.get("score"), 0.0),
        "rerank_score": r.get("rerank_score"),
        "chunk_id": _first(r.get("chunk_id"), c.get("chunk_id"), ""),
        "source": _first(r.get("source"), c.get("source"), "未知文件"),
        "section": _first(r.get("section"), c.get("section"), "其他"),
        "page": _first(r.get("page"), c.get("page"), "—"),
        "text": _first(r.get("text"), c.get("text"), ""),
    }


def _chunks_rows(retrieved: list) -> list:
    """
    检索命中 -> 明细表行（# / 检索分 / 重排分 / 来源 / 章节、页 / 内容预览）。

    「问答」与「检索」两个标签页共用；兼容扁平与嵌套两种命中形状，
    来源、章节、页码、文本直接来自分块元数据（chunking 阶段写入），
    不应出现问号占位。
    """
    rows = []
    for r in retrieved:
        m = _hit_meta(r)
        # 任务①：预览文本折叠空白字符（换行/连续空格），避免"看似空白"的预览
        text = " ".join(str(m["text"] or "").split())
        rscore = (f"{m['rerank_score']:.4f}"
                  if m.get("rerank_score") is not None else "—")
        rows.append([
            str(m["rank"]),
            f"{float(m['score']):.4f}",
            rscore,
            str(m["source"]),
            f"{m['section']} p{m['page']}",
            (text[:60] + "…") if len(text) > 60 else text,
        ])
    return rows


def build_gradio_ui():
    import gradio as gr

    papers = list_papers()
    names = [p["filename"] for p in papers]

    with gr.Blocks(title="学术文献 RAG 研读助手") as demo:
        gr.Markdown(
            "# 📚 学术文献 RAG 研读助手\n"
            "预生成结构化笔记优先，RAG 问答用于深挖原文细节。"
            "回答均基于检索到的文献片段，可溯源到 论文-章节-页码。")

        with gr.Tabs():
            # ---------------- Tab 1: 文献库 ----------------
            with gr.Tab("📖 文献库"):
                table = gr.Dataframe(
                    headers=["文件名", "标题", "页数", "已解析", "摘要"],
                    datatype=["str", "str", "str", "str", "str"],
                    value=_table_rows(papers), interactive=False, wrap=True)
                # --- 网页上传闭环：多选PDF -> data/papers -> 解析分块 -> 增量追加索引 ---
                with gr.Accordion("📤 上传新文献（PDF，支持多选）", open=False):
                    upload_btn = gr.File(
                        label="选择PDF文件（可多选）",
                        file_count="multiple",
                        file_types=[".pdf"],
                        type="filepath",
                    )
                    upload_submit = gr.Button("⬆️ 上传并加入知识库", variant="primary")
                with gr.Row():
                    refresh_btn = gr.Button("🔄 刷新列表")
                    delete_btn = gr.Button("🗑️ 删除选中文献", variant="stop")
                delete_status = gr.Textbox(label="删除状态", interactive=False)
                with gr.Row():
                    paper_dd = gr.Dropdown(choices=names, label="选择文献",
                                           interactive=True)
                    view_btn = gr.Button("📄 查看摘要")
                    gen_btn = gr.Button("✨ 生成摘要", variant="primary")
                summary_md = gr.Markdown(
                    "*(在上方选择文献即可查看摘要；尚未生成时可点击「生成摘要」)*")
                status_tb = gr.Textbox(label="状态", interactive=False)

            # ---------------- Tab 2: 问答 ----------------
            with gr.Tab("💬 问答"):
                with gr.Row():
                    with gr.Column(scale=1):
                        qa_dd = gr.Dropdown(
                            choices=[ALL_PAPERS] + names, value=ALL_PAPERS,
                            label="提问范围（选单篇 = 仅针对该文献解答）")
                        qa_topk = gr.Slider(4, 30, value=12, step=1,
                                            label="top_k（最终送入大模型的分块数）")
                        qa_rerank = gr.Checkbox(
                            value=True,
                            label="启用 Reranker 重排序（先取18块候选重排，再取top_k）")
                        qa_btn = gr.Button("🚀 提问", variant="primary")
                    with gr.Column(scale=3):
                        qa_question = gr.Textbox(
                            label="问题", lines=2,
                            placeholder="例：这篇论文的研究方法是什么？\n"
                                        "选「全部文献」时会逐篇解答，不会遗漏短篇。")
                qa_out = gr.Markdown("*(回答、溯源片段与告警信息将展示在这里)*")
                # 检索片段明细：展开可查看本次送入大模型的全部分块与相似度分数
                with gr.Accordion(
                        "🔎 检索片段明细（本次送入大模型的分块与相似度分数，"
                        "用于排查召回是否正确）", open=False):
                    chunks_df = gr.Dataframe(
                        headers=["#", "检索分", "重排分", "来源", "章节/页", "内容预览"],
                        datatype=["str", "str", "str", "str", "str", "str"],
                        value=[], interactive=False, wrap=True)

            # ---------------- Tab 3: 检索 ----------------
            with gr.Tab("🔍 检索"):
                gr.Markdown(
                    "直接查看检索结果与相似度分数（**不调用大模型、不消耗额度**），"
                    "用于排查召回质量；回答式提问请用「💬 问答」。")
                with gr.Row():
                    with gr.Column(scale=1):
                        rt_dd = gr.Dropdown(
                            choices=[ALL_PAPERS] + names, value=ALL_PAPERS,
                            label="检索范围")
                        rt_topk = gr.Slider(4, 30, value=10, step=1,
                                            label="返回条数")
                        rt_rerank = gr.Checkbox(
                            value=True, label="启用 Reranker 重排序")
                        rt_btn = gr.Button("🔍 检索", variant="primary")
                    with gr.Column(scale=3):
                        rt_query = gr.Textbox(
                            label="查询词", lines=2,
                            placeholder="例：小波变换 阈值处理")
                rt_out = gr.Dataframe(
                    headers=["#", "检索分", "重排分", "来源", "章节/页", "内容预览"],
                    datatype=["str", "str", "str", "str", "str", "str"],
                    value=[], interactive=False, wrap=True)

            # ---------------- Tab 4: 对比 ----------------
            with gr.Tab("⚖️ 对比"):
                gr.Markdown(
                    "勾选 ≥2 篇文献后对比：**填写问题**则围绕问题逐篇检索对比；"
                    "**问题留空**则自动执行全文内容对比（研究背景/研究方法/数据集/"
                    "核心结论/局限性五维度）。报告保存到 literature_lib/compare/。")
                cmp_question = gr.Textbox(
                    label="对比问题（留空 = 自动五维度对比）", lines=2,
                    placeholder="例：各篇文献采用了哪些研究方法？结论有何异同？\n"
                                "（留空则自动对比研究背景/方法/数据集/结论/局限性）")
                cmp_dd = gr.Dropdown(choices=names, multiselect=True,
                                     label="勾选参与对比的文献（至少2篇）")
                with gr.Row():
                    cmp_refresh = gr.Checkbox(
                        value=False, label="结合各篇已保存的结构化摘要一起对比")
                    cmp_btn = gr.Button("⚖️ 生成对比", variant="primary")
                cmp_out = gr.Markdown("*(对比结果将展示在这里)*")
                cmp_status = gr.Textbox(label="状态", interactive=False)

        # ---------------- 事件处理器（只调服务层） ----------------
        def refresh_all():
            papers = list_papers(only_indexed=True)
            return (_table_rows(papers),
                    gr.update(choices=[p["filename"] for p in papers]))

        def load_summary(filename):
            if not filename:
                return "*(请先选择文献)*", ""
            r = get_summary(filename)
            if not r["exists"]:
                return (f"*《{filename}》尚未生成摘要，点击上方「✨ 生成摘要」创建。*",
                        f"暂无摘要: {r['path']}")
            return r["content"], f"已加载: {r['path']}"

        def do_generate(filename):
            if not filename:
                raise gr.Error("请先在上方选择文献")
            try:
                res = generate_summary_for(filename)
            except Exception as e:
                raise gr.Error(f"生成失败: {e}")
            papers = list_papers()
            # 返回: 摘要内容 / 状态 / 刷新后的表格与下拉框
            return (res["markdown"], f"✅ 摘要已保存: {res['path']}",
                    _table_rows(papers),
                    gr.update(choices=[p["filename"] for p in papers]))

        def do_ask(question, source, top_k, rerank):
            if not question or not question.strip():
                raise gr.Error("请输入问题")
            src = None if (not source or source == ALL_PAPERS) else source
            try:
                res = ask_question(question.strip(), source=src, top_k=int(top_k),
                                   rerank=bool(rerank))
            except Exception as e:
                raise gr.Error(f"问答失败: {e}")
            # 复用统一的 Markdown 组装（含溯源、覆盖统计、零召回告警）
            md = summarizer.build_qa_markdown(
                res, metadata_filter={"source": src} if src else None)
            return md, _chunks_rows(res.get("retrieved", []))

        def do_retrieve(query, source, top_k, rerank):
            if not query or not query.strip():
                raise gr.Error("请输入查询词")
            src = None if (not source or source == ALL_PAPERS) else source
            try:
                res = run_retrieve(query.strip(), source=src, top_k=int(top_k),
                                   rerank=bool(rerank))
            except Exception as e:
                raise gr.Error(f"检索失败: {e}")
            return _chunks_rows(res["hits"])

        def do_compare(question, sources, refresh):
            # 边界判断：勾选不足2篇 → 友好提示；问题为空 → 自动对比模式
            if not sources:
                raise gr.Error("请先勾选至少两篇文献再对比")
            if len(sources) < 2:
                raise gr.Error("对比至少需要勾选两篇文献")
            q = (question or "").strip()
            if not q:
                print("[webapp] 对比问题为空 -> 自动对比模式（五维度全文内容对比）")
            try:
                res = run_compare_viewpoints(q, list(sources),
                                             use_saved_summaries=bool(refresh))
            except ValueError as e:  # 服务层边界错误（如文献不在索引中）
                raise gr.Error(str(e))
            except Exception as e:
                raise gr.Error(f"对比失败: {e}")
            mode = "自动对比" if res.get("mode") == "auto" else "问题对比"
            status = f"✅ {mode}报告已保存: {res['path']}"
            if res.get("failed"):
                status += f" | ⚠️ 生成失败: {', '.join(f['source'] for f in res['failed'])}"
            return res["markdown"], status

        def do_delete(filename):
            if not filename:
                raise gr.Error("请先选择要删除的文献")
            try:
                res = delete_paper(filename)
            except Exception as e:
                raise gr.Error(f"删除失败: {e}")
            papers = list_papers(only_indexed=True)
            choices = [p["filename"] for p in papers]
            status = f"✅ {res['detail']}"
            if not res["deleted"]:
                status = f"ℹ️ {res['detail']}"
            return (_table_rows(papers),
                    gr.update(choices=choices, value=choices[0] if choices else None),
                    gr.update(choices=[ALL_PAPERS] + choices),
                    gr.update(choices=[ALL_PAPERS] + choices),
                    gr.update(choices=choices),
                    status)

        def do_upload(files):
            """上传处理器：闭环入库 + 文献库/问答/对比下拉框全量联动刷新。"""
            if not files:
                raise gr.Error("请先选择至少一个PDF文件")
            # gr.File(filepath 模式)返回路径字符串或其列表
            if isinstance(files, (str, Path)):
                files = [files]
            try:
                res = ingest_uploaded_files([str(f) for f in files])
            except Exception as e:
                raise gr.Error(f"上传处理失败: {e}")
            # 状态汇总（逐文件明细 + 总计数），供页面提示
            lines = []
            for r in res["results"]:
                icon = {"ok": "✅", "duplicate": "⏭️", "invalid": "❌",
                        "broken": "❌", "failed": "❌"}.get(r["status"], "•")
                lines.append(f"{icon} {r['file']}: {r['detail']}")
            lines.append(f"合计: 入库 {res['n_ok']} / 跳过 {res['n_skip']} / "
                         f"失败 {res['n_fail']}，新增分块 {res['n_added_chunks']} 个"
                         "（旧索引未重建，增量追加）")
            summary_text = "\n".join(lines)
            print(f"[webapp] 上传完成: ok={res['n_ok']} skip={res['n_skip']} "
                  f"fail={res['n_fail']} added={res['n_added_chunks']}")
            # 下拉框联动：问答/检索/对比的范围立即包含新文献
            choices = res["dropdown_choices"]
            return (summary_text, res["table_rows"],
                    gr.update(choices=choices),    # 文献库-单篇查看
                    gr.update(choices=[ALL_PAPERS] + choices),  # 问答范围
                    gr.update(choices=[ALL_PAPERS] + choices),  # 检索范围
                    gr.update(choices=choices),    # 对比勾选
                    gr.update(value=[] if (res["n_fail"] == 0 and res["n_ok"] > 0) else files))  # 成功清空上传框（空列表），失败保留

        delete_btn.click(do_delete, inputs=[paper_dd],
                         outputs=[table, paper_dd, qa_dd, rt_dd, cmp_dd, delete_status])
        refresh_btn.click(refresh_all, outputs=[table, paper_dd])
        paper_dd.change(load_summary, inputs=[paper_dd],
                        outputs=[summary_md, status_tb])
        view_btn.click(load_summary, inputs=[paper_dd],
                       outputs=[summary_md, status_tb])
        gen_btn.click(do_generate, inputs=[paper_dd],
                      outputs=[summary_md, status_tb, table, paper_dd])
        qa_btn.click(do_ask, inputs=[qa_question, qa_dd, qa_topk, qa_rerank],
                     outputs=[qa_out, chunks_df])
        rt_btn.click(do_retrieve, inputs=[rt_query, rt_dd, rt_topk, rt_rerank],
                     outputs=[rt_out])
        cmp_btn.click(do_compare, inputs=[cmp_question, cmp_dd, cmp_refresh],
                      outputs=[cmp_out, cmp_status])
        upload_submit.click(
            do_upload, inputs=[upload_btn],
            outputs=[status_tb, table, paper_dd, qa_dd, rt_dd, cmp_dd, upload_btn])
        demo.load(refresh_all, outputs=[table, paper_dd])  # 页面打开时初始化

    return demo


# ==========================================================================
# 应用工厂：FastAPI 后端 + 挂载 Gradio 前端（同一进程、同一端口）
# ==========================================================================
def create_app() -> FastAPI:
    ensure_dirs()
    app = FastAPI(
        title="学术文献 RAG 研读助手 API",
        description="复用 CLI 同一套底层函数：文献列表 / 摘要生成与查看 / 问答溯源",
        version="0.1.0",
    )
    _register_api(app)
    demo = build_gradio_ui()
    demo.queue()  # 生成摘要/问答耗时较长，启用排队
    return gr_mount(app, demo)


def gr_mount(app: FastAPI, demo):
    """把 Gradio 页面挂载到 FastAPI 根路径（延迟导入 gradio）。

    Gradio 6 起 theme 等参数从 Blocks 构造器移到了 launch/mount，
    这里统一在挂载时传入。
    """
    import gradio as gr
    try:
        return gr.mount_gradio_app(app, demo, path="/", theme=gr.themes.Soft())
    except TypeError:  # 兼容旧版本：theme 仍属于 Blocks 构造参数
        return gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    setup_console()
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=7860)
