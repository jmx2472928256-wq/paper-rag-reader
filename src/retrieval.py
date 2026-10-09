# -*- coding: utf-8 -*-
"""
混合检索模块：BM25 稀疏检索 + FAISS 稠密向量检索 + 分数融合。

设计
----
1. BM25（rank_bm25.BM25Okapi）：关键词精确匹配能力强，对公式符号、
   专有名词、数字等长尾词敏感；分词用 jieba（中文）+ 兜底正则（英文）；
2. FAISS（IndexFlatIP）：语料向量经 L2 归一化后用内积检索，等价于余弦
   相似度；向量来自智谱 embedding-3；
3. 融合策略：
   - weighted（默认）：BM25 分数做 min-max 归一化后与稠密余弦分数加权，
     score = alpha * dense + (1 - alpha) * bm25，alpha 可调；
   - rrf：倒数排名融合 Reciprocal Rank Fusion，score = Σ 1/(60 + rank)，
     无需分数标定，对两路量纲差异更鲁棒。
4. 支持三种检索模式：bm25 / dense / hybrid（默认），供消融实验切换。

索引持久化
----------
data/index/
  faiss.index       FAISS 索引
  embeddings.npy    语料向量缓存（重复构建时免二次调用 embedding API）
  meta.json         元信息（模型名、维度、chunk 文件路径、构建时间）

用法
----
    python -m src.retrieval                 # 构建/重建索引
    python -m src.retrieval "query text"    # 构建后交互式试查
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from .config import (
        CHUNKS_FILE, DATA_PAPERS_DIR, EMBEDDING_DIM, EMBEDDING_MODEL, EMB_FILE,
        FAISS_FILE, INDEX_DIR, META_FILE, PARSED_DIR, PAPERS_DIR,
        ensure_dirs, setup_console,
    )
    from . import chunking
except ImportError:  # 以脚本方式直接运行
    from config import (
        CHUNKS_FILE, DATA_PAPERS_DIR, EMBEDDING_DIM, EMBEDDING_MODEL, EMB_FILE,
        FAISS_FILE, INDEX_DIR, META_FILE, PARSED_DIR, PAPERS_DIR,
        ensure_dirs, setup_console,
    )
    import chunking

import faiss
from rank_bm25 import BM25Okapi

# jieba 为可选依赖：缺失时退化为简单正则分词（英文可用，中文效果差）
try:
    import jieba
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False

# 中英文停用词表（精简版，主要过滤高频虚词以提升 BM25 区分度）
_STOPWORDS = {
    "the", "a", "an", "of", "and", "to", "in", "is", "are", "for", "on",
    "with", "as", "by", "we", "our", "this", "that", "be", "it", "from",
    "at", "or", "not", "can", "will", "为", "的", "了", "和", "是", "在",
    "与", "及", "对", "等", "中", "上", "下", "或", "其", "也", "并",
}


# ==========================================================================
# 分词
# ==========================================================================
def tokenize(text: str) -> list:
    """
    混合语料分词：安装 jieba 时用其对中英文统一分词；否则用正则抽取
    英文单词/数字与连续中文串。返回小写化、去停用词后的 token 列表。
    """
    text = (text or "").lower()
    if _HAS_JIEBA:
        tokens = jieba.lcut(text)
    else:
        # \w+ 可覆盖英文单词、数字与连续中文串（连续汉字会粘连成整段，
        # 仅作兜底，建议安装 jieba）
        import re
        tokens = re.findall(r"\w+", text)
    return [t for t in tokens if t.strip() and t not in _STOPWORDS
            and any(ch.isalnum() for ch in t)]


def _normalize_scores(scores) -> np.ndarray:
    """min-max 归一化到 [0, 1]；全零/常量序列返回均匀值，避免除零。"""
    arr = np.asarray(scores, dtype="float32")
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.full_like(arr, 0.5 if hi > 0 else 0.0)
    return (arr - lo) / (hi - lo)


def _match_metadata(chunk: dict, metadata_filter=None) -> bool:
    """
    判断分块是否满足元数据过滤条件（检索的 filter 参数）。

    metadata_filter 形如:
        {"source": "paper1.pdf"}                       # 只检索某个PDF
        {"source": ["a.pdf", "b.pdf"]}                 # 多个PDF任一命中
        {"source": "paper1.pdf", "section": "method"}  # 多key之间为AND
    值为列表/元组/集合表示"命中其一即可"；None 或空 dict 表示不过滤。
    """
    if not metadata_filter:
        return True
    for key, allowed in metadata_filter.items():
        val = chunk.get(key)
        if isinstance(allowed, (list, tuple, set)):
            if val not in allowed:
                return False
        elif val != allowed:
            return False
    return True


# ==========================================================================
# 混合检索器
# ==========================================================================
class HybridRetriever:
    """BM25 + FAISS 双路检索器（可在检索时选择模式与融合策略）。"""

    def __init__(self, chunks: list, dim: int = EMBEDDING_DIM):
        if not chunks:
            raise ValueError("HybridRetriever 需要非空的 chunk 列表。")
        self.chunks = chunks
        self.dim = dim
        # ---- BM25 通道：建一次即可（纯内存计算，代价低） ----
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        # ---- 稠密通道：向量矩阵与 FAISS 索引，由 build_dense / load 填充 ----
        self.faiss_index = None
        self._emb_matrix = None

    # ---------------- 稠密索引构建 ----------------
    def build_dense(self, embeddings=None) -> None:
        """
        构建稠密向量索引。

        Args:
            embeddings: 可选的外部向量矩阵（如磁盘缓存）；为 None 时
                        调用智谱 embedding-3 在线向量化全部 chunk。
        """
        if embeddings is None:
            texts = [c["text"] for c in self.chunks]
            print(f"[retrieval] 调用 {EMBEDDING_MODEL} 向量化 {len(texts)} 个分块 ...")
            t0 = time.time()
            embeddings = _embed_with_progress(texts)
            print(f"[retrieval] 向量化完成，耗时 {time.time() - t0:.1f}s")
        self._set_embeddings(embeddings)

    def _set_embeddings(self, embeddings) -> None:
        """用给定向量矩阵初始化 FAISS 索引（构建与加载共用）。"""
        arr = np.ascontiguousarray(np.asarray(embeddings, dtype="float32"))
        if arr.shape[0] != len(self.chunks):
            raise ValueError(f"向量数量({arr.shape[0]})与 chunk 数({len(self.chunks)})不一致")
        faiss.normalize_L2(arr)  # L2 归一化 -> IndexFlatIP 内积即余弦相似度
        self._emb_matrix = arr
        self.faiss_index = faiss.IndexFlatIP(arr.shape[1])
        self.faiss_index.add(arr)

    # ---------------- 检索 ----------------
    def search(self, query: str, top_k: int = 5, mode: str = "hybrid",
               fusion: str = "weighted", alpha: float = 0.5,
               allowed_sections=None, metadata_filter=None,
               query_vec=None) -> list:
        """
        混合检索。

        Args:
            query: 查询文本（自然语言问题）。
            top_k: 返回的块数。
            mode:  "bm25" / "dense" / "hybrid"。
            fusion: "weighted"（加权融合）或 "rrf"（倒数排名融合）。
            alpha: weighted 融合下稠密分数权重（0~1）。
            allowed_sections: 仅允许的规范章节名集合（如排除 references）；
                              None 表示不过滤。
            metadata_filter: 元数据过滤参数（filter），按分块 metadata 过滤，
                如 {"source": "paper1.pdf"} 只检索指定 PDF、
                {"source": ["a.pdf", "b.pdf"]} 多文件任一命中、
                {"source": "...", "section": "method"} 多条件 AND。
                None 表示不过滤。可与 allowed_sections 叠加。
            query_vec: 预计算并 L2 归一化的查询向量（供 search_balanced 多路
                检索复用同一向量，避免每篇重复调用 embedding API）；
                None 则内部计算。

        Returns:
            [{"rank", "chunk_id", "score", "chunk"}, ...] 按融合分数降序。
        """
        # 任务⑤：top_k 防御性边界保护（非法输入钳制，防止崩溃）
        try:
            top_k = max(1, int(top_k))
        except (TypeError, ValueError):
            top_k = 5
        n = len(self.chunks)

        # ---- 元数据预过滤：先算出允许参与排序的分块下标集合 ----
        # BM25 对全库打分后在子集内排序不漏召回；稠密通道带过滤时改为在
        # 子集上做精确余弦（FAISS 候选池可能漏掉小子集内的命中）。
        keep = None
        if allowed_sections is not None or metadata_filter:
            keep = {i for i, c in enumerate(self.chunks)
                    if (allowed_sections is None or c.get("section") in allowed_sections)
                    and _match_metadata(c, metadata_filter)}
            if not keep:  # 过滤条件过严：直接返回空
                return []

        cand_k = min(max(top_k * 4, 20), n)

        # ---- 通道 1：BM25（关键词精确匹配）----
        bm25_scores = {}
        if mode in ("bm25", "hybrid"):
            scores = _normalize_scores(self.bm25.get_scores(tokenize(query)))
            for i in np.argsort(-scores):  # 全库按分数降序
                if scores[i] <= 1e-6:  # 后续更低，全部剪枝
                    break
                if keep is not None and int(i) not in keep:
                    continue
                bm25_scores[int(i)] = float(scores[i])
                if len(bm25_scores) >= cand_k:
                    break

        # ---- 通道 2：稠密向量（语义相似度）----
        dense_scores = {}
        if mode in ("dense", "hybrid") and self.faiss_index is not None:
            q_vec = query_vec if query_vec is not None else self._embed_query(query)
            if keep is None:
                # 无过滤：FAISS 内积检索（向量已 L2 归一化，内积即余弦）
                dists, idxs = self.faiss_index.search(q_vec, cand_k)
                pairs = [(float(s), int(i)) for s, i in zip(dists[0], idxs[0]) if i != -1]
            else:
                # 带过滤：在过滤子集上精确计算余弦（矩阵已归一化，点积即可）
                sub = sorted(keep)
                sims = self._emb_matrix[sub] @ q_vec.reshape(-1)
                top = np.argsort(-sims)[:cand_k]
                pairs = [(float(sims[j]), sub[j]) for j in top]
            # 余弦可能为负（罕见），截断到 [0, 1] 与 BM25 分数量纲对齐
            for score, idx in pairs:
                dense_scores[idx] = max(0.0, min(1.0, score))

        # ---- 融合 ----
        if fusion == "rrf":
            # 倒数排名融合：每路按名次贡献 1/(60+rank)，名次越靠前贡献越大
            fused = {}
            for channel in (bm25_scores, dense_scores):
                for rank, (i, _) in enumerate(
                        sorted(channel.items(), key=lambda kv: kv[1], reverse=True)):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (60.0 + rank + 1)
        elif mode == "bm25":
            fused = dict(bm25_scores)
        elif mode == "dense":
            fused = dict(dense_scores)
        else:  # weighted：对两路分数加权求和（未命中的通道记 0 分）
            fused = {
                i: alpha * dense_scores.get(i, 0.0) + (1.0 - alpha) * bm25_scores.get(i, 0.0)
                for i in set(bm25_scores) | set(dense_scores)
            }

        # ---- 排序 + 元数据过滤兜底 ----
        results = []
        for i, score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
            if keep is not None and i not in keep:
                continue
            chunk = self.chunks[i]
            results.append({
                "rank": len(results) + 1,
                "chunk_id": chunk["chunk_id"],
                "score": round(float(score), 4),
                "chunk": chunk,
            })
            if len(results) >= top_k:
                break
        return results

    # ---------------- 查询向量化 ----------------
    def _embed_query(self, query: str):
        """查询文本向量化（L2 归一化），供稠密通道使用。"""
        q_vec = np.asarray(_embed_with_progress([query])[0], dtype="float32").reshape(1, -1)
        faiss.normalize_L2(q_vec)
        return q_vec

    def distinct_sources(self) -> list:
        """索引中全部非空的原始 PDF 文件名（去重排序）。"""
        return sorted({c.get("source") for c in self.chunks if c.get("source")})

    # ---------------- 按文献均衡召回 ----------------
    def search_balanced(self, query: str, top_k: int = 12, mode: str = "hybrid",
                        fusion: str = "weighted", alpha: float = 0.5,
                        allowed_sections=None, sources=None,
                        query_vec=None) -> list:
        """
        按文献均衡召回：解决全库/多文档查询时短文档被长文档整体挤出
        top_k 的问题。

        算法（不改变单路检索逻辑，只做结果层配额合并）：
        1. 对每篇目标文献独立调用 search()（metadata_filter 限定该篇，
           分数在全库口径下计算，跨篇可比）；
        2. 保底阶段：每篇按 quota = max(1, top_k // 文献数) 取头部块，
           保证任何一篇（只要有相关片段）都不会被整体忽略；
        3. 补足阶段：剩余名额由各篇按其内部排名轮转竞争（好片段优先），
           长文档的高分片段仍可多占名额。

        Args:
            sources: 目标文献文件名列表；None 表示索引中全部文献。
            query_vec: 预计算的归一化查询向量，全部文献共用一次 embedding。
            其余参数同 search()。

        Returns:
            与 search() 相同结构的命中列表（按融合分数降序，rank 已重排），
            数量不超过 top_k。注意: top_k 小于文献数时无法保证每篇入选，
            建议 top_k >= 文献数。
        """
        # 任务⑤：top_k 防御性边界保护（非法输入钳制，防止崩溃）
        try:
            top_k = max(1, int(top_k))
        except (TypeError, ValueError):
            top_k = 12
        all_sources = self.distinct_sources()
        targets = [s for s in (sources or all_sources) if s]
        if not targets:
            return []
        # 单篇无需均衡，退化为普通检索
        if len(targets) == 1:
            return self.search(query, top_k=top_k, mode=mode, fusion=fusion,
                               alpha=alpha, allowed_sections=allowed_sections,
                               metadata_filter={"source": targets[0]},
                               query_vec=query_vec)

        # 全部文献共用一次查询向量化（稠密通道需要时才计算）
        if query_vec is None and mode in ("dense", "hybrid"):
            query_vec = self._embed_query(query)

        per_source = []
        for s in targets:
            hits = self.search(query, top_k=top_k, mode=mode, fusion=fusion,
                               alpha=alpha, allowed_sections=allowed_sections,
                               metadata_filter={"source": s}, query_vec=query_vec)
            per_source.append(hits)

        # ---- 阶段1：每篇保底配额 ----
        quota = max(1, top_k // len(targets))
        picked, used = [], set()
        for hits in per_source:
            for h in hits[:quota]:
                if h["chunk_id"] in used:
                    continue
                used.add(h["chunk_id"])
                picked.append(h)

        # ---- 阶段2：剩余名额按各篇内部排名轮转补足（好片段优先）----
        remain = top_k - len(picked)
        pools = [list(hits[quota:]) for hits in per_source]
        while remain > 0 and any(pools):
            progressed = False
            for pool in pools:
                if remain <= 0:
                    break
                while pool and pool[0]["chunk_id"] in used:
                    pool.pop(0)
                if pool:
                    h = pool.pop(0)
                    used.add(h["chunk_id"])
                    picked.append(h)
                    remain -= 1
                    progressed = True
            if not progressed:
                break

        # 统一按融合分数降序输出并重排名次
        picked.sort(key=lambda h: h["score"], reverse=True)
        picked = picked[:top_k]
        for rank, h in enumerate(picked, 1):
            h["rank"] = rank
        return picked

    # ---------------- 持久化 ----------------
    def save(self, index_dir=INDEX_DIR, chunks_file: Path = CHUNKS_FILE) -> None:
        """
        保存索引到 data/index/：
        faiss.index + embeddings.npy（免重复调用向量化 API）+ meta.json。
        chunk 语料本体保存在 chunks_file（data/chunks/chunks.json），
        meta.json 记录其绝对路径以便 load 时对齐。
        """
        if self.faiss_index is None:
            raise RuntimeError("稠密索引尚未构建，请先调用 build_dense()。")
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(index_dir / "faiss.index"))
        np.save(index_dir / "embeddings.npy", self._emb_matrix)
        meta = {
            "embedding_model": EMBEDDING_MODEL,
            "dim": self.dim,
            "num_chunks": len(self.chunks),
            "chunks_file": str(Path(chunks_file).resolve()),
            # 分块清单（与向量行一一对应）：增量构建时按 chunk_id+文本md5
            # 精确复用未变更分块的向量，避免重复调用 embedding API
            "chunk_ids": [c["chunk_id"] for c in self.chunks],
            "chunk_md5s": [_md5(c.get("text", "")) for c in self.chunks],
            "built_at": datetime.now().isoformat(timespec="seconds"),
        }
        (index_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[retrieval] 索引已保存 -> {index_dir}")

    @classmethod
    def load(cls, index_dir=INDEX_DIR) -> "HybridRetriever":
        """从磁盘加载索引（复用向量缓存，不重新调用 embedding API）。"""
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        chunks = json.loads(Path(meta["chunks_file"]).read_text(encoding="utf-8"))
        retriever = cls(chunks, dim=meta.get("dim", EMBEDDING_DIM))
        embeddings = np.load(index_dir / "embeddings.npy")
        retriever._set_embeddings(embeddings)
        return retriever


def _embed_with_progress(texts: list) -> list:
    """带重试/截断的批量向量化（复用 config.embed_texts 的统一实现）。"""
    try:
        from .config import embed_texts
    except ImportError:  # 以脚本方式直接运行
        from config import embed_texts
    return embed_texts(texts)


# ==========================================================================
# 索引构建入口
# ==========================================================================
def _md5(text: str) -> str:
    """分块文本的内容哈希（向量缓存复用的一致性校验）。"""
    import hashlib
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def _load_vector_cache(index_dir=INDEX_DIR):
    """
    读取旧索引的向量缓存，返回 {(chunk_id, 文本md5): 向量} 映射。

    复用条件同时校验 chunk_id 与分块文本哈希：即使解析/分块规则升级
    导致文本变化，也绝不会把旧向量错配到新文本上（避免向量与内容脱节）。
    优先使用 meta.json 中保存的 chunk_ids/chunk_md5s 清单（与向量行一一
    对齐，不依赖会被重建覆盖的 chunks.json 文件——分块数量/文本变化时
    缓存依然可精确命中未变更的分块）；读取失败时返回空 dict（回退为
    全量重新向量化）。
    """
    cache = {}
    meta_path = Path(index_dir) / "meta.json"
    emb_path = Path(index_dir) / "embeddings.npy"
    if not (meta_path.exists() and emb_path.exists()):
        return cache
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        old_emb = np.load(emb_path)
        ids, md5s = meta.get("chunk_ids"), meta.get("chunk_md5s")
        if (ids is not None and md5s is not None
                and len(ids) == len(md5s) == old_emb.shape[0]):
            # 新版 meta：清单与向量行严格对齐
            for i, cid in enumerate(ids):
                cache[(cid, md5s[i])] = old_emb[i]
            return cache
        # 旧版 meta 兼容：回退用 chunks.json 配对（仅数量一致时可靠）
        old_chunks = json.loads(Path(meta["chunks_file"]).read_text(encoding="utf-8"))
        if len(old_chunks) == old_emb.shape[0]:
            for i, c in enumerate(old_chunks):
                cache[(c["chunk_id"], _md5(c.get("text", "")))] = old_emb[i]
    except Exception as e:
        print(f"[retrieval] 旧向量缓存读取失败，将全量重新向量化: {e}")
        return {}
    return cache


def build_index(parsed_dir=PARSED_DIR, chunks_file=CHUNKS_FILE,
                index_dir=INDEX_DIR) -> HybridRetriever:
    """
    端到端构建索引：增量解析缺失的 PDF -> 多粒度分块 -> 向量化 ->
    保存 FAISS 索引与元信息。幂等：语料变化时重跑即可。

    向量化按需增量执行：旧索引中 chunk_id 与文本哈希都一致的分块直接
    复用缓存向量，只有新增/变更的分块才调用 embedding API，避免重复
    消耗额度。
    """
    ensure_dirs()
    parsed_files = sorted(Path(parsed_dir).glob("*.json"))
    if not parsed_files:
        pdfs = (list(Path(PAPERS_DIR).glob("*.pdf"))
                + list(Path(PAPERS_DIR).glob("*.PDF"))
                + list(Path(DATA_PAPERS_DIR).glob("*.pdf"))    # 网页上传目录
                + list(Path(DATA_PAPERS_DIR).glob("*.PDF")))
        if not pdfs:
            raise FileNotFoundError(
                "papers/ 目录下没有 PDF，data/parsed/ 也没有解析结果，无法构建索引。\n"
                "请先将论文 PDF 放入 papers/ 目录。")
        print("[retrieval] 未发现解析结果，先执行 PDF 解析 ...")
        try:
            from .pdf_parser import batch_parse
        except ImportError:  # 以脚本方式直接运行
            from pdf_parser import batch_parse
        batch_parse()
        parsed_files = sorted(Path(parsed_dir).glob("*.json"))
        if not parsed_files:
            raise RuntimeError("PDF 解析后仍未得到结果，无法构建索引。")

    chunks = chunking.build_all_chunks(parsed_dir, chunks_file)
    retriever = HybridRetriever(chunks)

    # ---- 增量向量化：能复用的复用，只对新增/变更分块调用 API ----
    cache = _load_vector_cache(index_dir)
    need = [c for c in chunks if (c["chunk_id"], _md5(c["text"])) not in cache]
    reused_n = len(chunks) - len(need)
    if need:
        print(f"[retrieval] 向量化 {len(need)} 个新/变更分块"
              f"（复用缓存 {reused_n} 个，节省API额度）...")
        fresh = _embed_with_progress([c["text"] for c in need])
        for c, vec in zip(need, fresh):
            cache[(c["chunk_id"], _md5(c["text"]))] = np.asarray(vec, dtype="float32")
    else:
        print(f"[retrieval] 全部 {len(chunks)} 个分块命中向量缓存，0 次API调用。")
    matrix = np.stack([cache[(c["chunk_id"], _md5(c["text"]))] for c in chunks]).astype("float32")
    retriever._set_embeddings(matrix)
    retriever.save(index_dir, chunks_file=chunks_file)
    return retriever


def _md5(text: str) -> str:
    """分块文本的内容哈希（向量缓存复用的一致性校验）。"""
    import hashlib
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def load_or_build_retriever(index_dir=INDEX_DIR, auto_build: bool = True) -> HybridRetriever:
    """
    加载已有索引；不存在且 auto_build=True 时自动走完整构建流程
    （问答/评估入口共用，保证"开箱即用"）。
    """
    index_dir = Path(index_dir)
    if (index_dir / "faiss.index").exists() and (index_dir / "meta.json").exists():
        print(f"[retrieval] 加载已有索引: {index_dir}")
        return HybridRetriever.load(index_dir)
    if not auto_build:
        raise FileNotFoundError(
            f"索引不存在 ({index_dir})，请先运行: python -m src.main index")
    return build_index(index_dir=index_dir)


def append_to_index(new_chunks: list, index_dir=INDEX_DIR,
                    chunks_file=CHUNKS_FILE) -> tuple:
    """
    把新增分块**增量追加**到现有 FAISS 索引（index.add 追加写入，
    不删除、不重建已有旧向量——网页上传闭环的核心原语）。

    流程：读磁盘现有索引/向量/分块清单 -> 过滤已存在的 chunk_id（幂等）->
    仅向量化新增分块 -> faiss_index.add 追加 -> 原子性落盘
    （faiss.index / embeddings.npy / chunks.json / meta.json 四件同步更新）。

    Args:
        new_chunks: 新文档的分块列表（由 chunking.chunk_paper 产出）。
        index_dir / chunks_file: 索引目录与分块文件路径。

    Returns:
        (retriever, n_added)：包含全部新旧分块、可直接挂回 RAGPipeline
        的检索器；本次实际向量化并追加的分块数。

    Raises:
        FileNotFoundError / RuntimeError: 磁盘索引不存在或状态不一致
        （分块数与向量行数不匹配）——此时不做任何写操作，由调用方决定
        是否回退全量重建。
    """
    index_dir = Path(index_dir)
    if not (index_dir / "faiss.index").exists():
        raise FileNotFoundError(
            f"索引不存在: {index_dir}。请先执行全量构建: python -m src.main index")

    index = faiss.read_index(str(index_dir / "faiss.index"))
    old_emb = np.load(index_dir / "embeddings.npy")
    old_chunks = json.loads(Path(chunks_file).read_text(encoding="utf-8"))
    # 状态一致性检查：任一不匹配都拒绝追加，防止写坏索引
    if not (len(old_chunks) == old_emb.shape[0] == index.ntotal):
        raise RuntimeError(
            f"磁盘索引状态不一致（分块{len(old_chunks)}/向量{old_emb.shape[0]}"
            f"/FAISS{index.ntotal}），拒绝增量追加。"
            "请人工核对 data/ 目录，或执行全量重建: python -m src.main index")
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))

    # 幂等去重：chunk_id 已存在的直接跳过（重复上传/重复解析防护）
    existing_ids = {c["chunk_id"] for c in old_chunks}
    fresh = [c for c in new_chunks if c["chunk_id"] not in existing_ids]
    if not fresh:
        return HybridRetriever.load(index_dir), 0

    # 仅向量化新增分块并追加（normalize 后与既有向量同口径）
    print(f"[retrieval] 增量追加: 向量化 {len(fresh)} 个新分块（旧索引 "
          f"{index.ntotal} 条不动）...")
    new_vecs = np.ascontiguousarray(
        np.asarray(_embed_with_progress([c["text"] for c in fresh]),
                   dtype="float32"))
    faiss.normalize_L2(new_vecs)
    index.add(new_vecs)  # FAISS 原生追加，旧向量零改动

    all_chunks = old_chunks + fresh
    all_emb = np.ascontiguousarray(np.vstack([old_emb, new_vecs]).astype("float32"))

    # ---- 四件同步落盘（与全量构建的格式完全一致）----
    faiss.write_index(index, str(index_dir / "faiss.index"))
    np.save(index_dir / "embeddings.npy", all_emb)
    Path(chunks_file).write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    meta.update({
        "num_chunks": len(all_chunks),
        "chunk_ids": [c["chunk_id"] for c in all_chunks],
        "chunk_md5s": [_md5(c.get("text", "")) for c in all_chunks],
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    })
    (index_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 内存检索器：BM25 按全量分块重建（含新块）；FAISS 直接复用追加后的
    # 索引对象与向量矩阵，不做任何重建
    retriever = HybridRetriever(all_chunks, dim=meta.get("dim", EMBEDDING_DIM))
    retriever.faiss_index = index
    retriever._emb_matrix = all_emb
    print(f"[retrieval] 增量追加完成: 索引总量 {index.ntotal} 条"
          f"（新增 {len(fresh)}）")
    return retriever, len(fresh)


if __name__ == "__main__":
    import sys
    setup_console()
    if len(sys.argv) > 1:  # 简易试查：python -m src.retrieval "查询词"
        retriever = load_or_build_retriever()
        for hit in retriever.search(" ".join(sys.argv[1:]), top_k=5):
            c = hit["chunk"]
            print(f"[{hit['rank']}] {hit['score']:.4f} {c['chunk_id']} "
                  f"《{c['paper_title']}》-{c['section']}-p{c['page']}")
            print(f"    {c['text'][:100]}...")
    else:
        build_index()
