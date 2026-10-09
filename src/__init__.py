# -*- coding: utf-8 -*-
"""学术文献 RAG 研读助手 —— 核心算法包。

模块一览
--------
- config          全局配置、目录约定、智谱AI客户端（GLM-5.3 / embedding-3）
- pdf_parser      PDF 结构化解析：章节识别、页眉页脚去噪、批量处理
- chunking        多粒度分块（句子级 / 段落级 / 章节级）
- retrieval       混合检索：BM25 稀疏检索 + FAISS 稠密向量检索 + 融合排序
- rag_pipeline    RAG 问答链路：上下文组装、防幻觉 prompt、引用溯源
- paper_analysis  跨论文指标对比、参考文献提取、networkx 引用网络图
- evaluator       评估：召回率、问答准确率、幻觉率、消融实验
- main            命令行入口 demo

使用方式
--------
    python -m src.main parse     # 解析 papers/ 下的 PDF
    python -m src.main index     # 构建混合检索索引
    python -m src.main ask "..." # 提问
    python -m src.main demo      # 端到端演示
"""

__all__ = [
    "config",
    "pdf_parser",
    "chunking",
    "retrieval",
    "rag_pipeline",
    "paper_analysis",
    "evaluator",
    "main",
]

__version__ = "0.1.0"
