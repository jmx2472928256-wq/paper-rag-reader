# -*- coding: utf-8 -*-
"""
轻量级本地 Reranker 重排序模块（零第三方依赖，不引入额外复杂依赖）。

定位
----
RAG 检索链路：混合召回（BM25 + FAISS）先取 RERANK_CANDIDATES 个候选，
本模块对候选做精细特征重打分，再取 top_k 送入大模型，提升最终证据质量。

重排特征（全部本地计算，取值均归一化到 [0, 1]）
------------------------------------------------
1. term_coverage  查询词覆盖率：查询分词（复用 retrieval.tokenize，jieba +
                   停用词过滤）后，出现在分块中的词占比——衡量主题相关性；
2. bigram_overlap 相邻词对（二元组）覆盖率：捕捉短语级精确匹配，比单词
                   更能区分"碰巧共词"与"真的在讲同一件事"；
3. prior          原混合检索融合分数（候选内 min-max 归一化）作为先验，
                   防止重排完全偏离检索排序（保守策略）。

rerank_score = W_TERM * coverage + W_BIGRAM * bigram + W_PRIOR * prior
"""

try:
    from .retrieval import tokenize
except ImportError:  # 以脚本方式直接运行
    from retrieval import tokenize

# 特征权重（和为 1；prior 权重最大以保证重排保守、可解释）
W_TERM = 0.45
W_BIGRAM = 0.25
W_PRIOR = 0.30


class LightweightReranker:
    """对召回候选按 精细词特征 + 检索先验 重打分的本地重排序器。"""

    def __init__(self, w_term: float = W_TERM, w_bigram: float = W_BIGRAM,
                 w_prior: float = W_PRIOR):
        self.w_term = w_term
        self.w_bigram = w_bigram
        self.w_prior = w_prior

    # ------------------------------------------------------------------
    def _features(self, query_terms: list, query_bigrams: set,
                  chunk_text: str) -> tuple:
        """计算 (term_coverage, bigram_coverage) 两个词级特征。"""
        c_terms = tokenize(chunk_text)
        if not query_terms:
            return 0.0, 0.0
        c_set = set(c_terms)
        q_set = set(query_terms)
        coverage = len(c_set & q_set) / len(q_set)
        bigram = 0.0
        if query_bigrams:
            c_bigrams = {(c_terms[i], c_terms[i + 1]) for i in range(len(c_terms) - 1)}
            bigram = len(query_bigrams & c_bigrams) / len(query_bigrams)
        return coverage, bigram

    # ------------------------------------------------------------------
    def rerank(self, query: str, hits: list, top_k: int = None) -> list:
        """
        对召回候选重打分并排序。

        Args:
            query: 用户查询文本。
            hits: 检索器返回的命中列表 [{"rank", "chunk_id", "score", "chunk"}, ...]。
            top_k: 截断数量；None 表示返回全部已排序候选（调用方可先取全量
                   打分结果再做逐篇覆盖保底等策略，最后自行截断）。

        Returns:
            新列表（不修改入参元素），每条附加 "rerank_score" 字段，按其降序；
            截断时 rank 已更新为 1..top_k。
        """
        if not hits:
            return []

        # prior 归一化基准（候选内 min-max）
        priors = [h.get("score", 0.0) for h in hits]
        lo, hi = min(priors), max(priors)
        span = (hi - lo) or 1.0

        q_terms = tokenize(query)
        q_bigrams = {(q_terms[i], q_terms[i + 1]) for i in range(len(q_terms) - 1)}

        scored = []
        for h in hits:
            coverage, bigram = self._features(q_terms, q_bigrams,
                                              h.get("chunk", {}).get("text", ""))
            prior = (h.get("score", 0.0) - lo) / span
            score = (self.w_term * coverage
                     + self.w_bigram * bigram
                     + self.w_prior * prior)
            h2 = dict(h)  # 复制条目，保留原 "score"（检索分），新增重排分
            h2["rerank_score"] = round(float(score), 4)
            scored.append(h2)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        if top_k is not None:
            scored = scored[:top_k]
            for rank, h in enumerate(scored, 1):
                h["rank"] = rank
        return scored
