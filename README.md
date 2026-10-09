# 学术文献 RAG 研读助手

基于 **检索增强生成（RAG）** 的学术文献研读工具：批量解析论文 PDF、多粒度分块、BM25 + FAISS 混合检索、受约束的 GLM-5.3 问答与**引用溯源**，并提供跨论文指标对比、引用网络分析与评估消融实验。

> 技术栈：Python · LlamaIndex · FAISS · pdfplumber · rank-bm25 · networkx · 智谱AI（GLM-5.3 / embedding-3）

## 功能特性

- **PDF 结构化解析**：识别 abstract / introduction / method / experiment / conclusion 等章节，自动去除页眉页脚噪声，记录每章页码区间，批量处理 `papers/` 目录
- **多粒度分块**：句子级 / 段落级 / 章节级三种粒度，兼顾精确检索与长上下文问答
- **混合检索**：BM25（关键词，jieba 分词）+ FAISS 稠密向量（embedding-3）双路召回，支持加权融合与 RRF 融合，可切换纯单路模式
- **可信问答**：Prompt 硬性约束"仅可参考提供的文献片段、知识库无相关信息时明确说明、禁止编造"，答案自动标注 `[n]` 引用并溯源到 **文档名称-章节-页码**；引用了不存在编号会被标记为幻觉信号
- **Reranker 重排序**：检索先取 18 个候选，经本地轻量重排（查询词覆盖率 + 二元组匹配 + 检索分先验，零第三方依赖）后取 top_k 送入大模型；开关可通过界面勾选 / `--no-rerank` / `.env: RERANK_ENABLED` 控制
- **检索片段明细**：网页问答可展开查看送入大模型的每个分块及其检索分/重排分，方便排查召回质量
- **分块模式可选**：`char` 字符滑窗（默认）与 `semantic` 语义分块（TextTiling 风格，零依赖，`python -m src.chunking --mode semantic`）；块间 overlap 默认为 chunk_size 的 15%
- **跨论文分析**：参考文献提取、库内引用网络图（networkx + matplotlib）、实验指标正则抽取与对比表
- **评估与消融**：Recall@k、问答准确率（关键词 / LLM 判卷）、幻觉率（LLM 事实核查 / 引用代理），检索配置消融实验一键出对比表

## 目录结构

```
wenxianzongjie/
├── papers/                 # ① 把要研读的论文 PDF 放这里
├── src/
│   ├── config.py           # 全局配置：.env 密钥加载、路径约定、智谱AI客户端
│   ├── pdf_parser.py       # PDF 结构化解析（章节识别/去噪/批量）
│   ├── chunking.py         # 多粒度分块（句子/段落/章节）
│   ├── retrieval.py        # 混合检索（BM25 + FAISS + 融合排序 + 元数据过滤）
│   ├── rag_pipeline.py     # RAG 问答链路、引用溯源、防幻觉约束
│   ├── summarizer.py       # 结构化摘要/多篇对比/问答Markdown组装
│   ├── paper_analysis.py   # 跨论文指标对比、参考文献、引用网络图
│   ├── evaluator.py        # 评估：召回率/准确率/幻觉率/消融实验
│   └── main.py             # 命令行入口
├── batch_run.py            # 批量任务脚本（摘要生成 + 问答）
├── literature_lib/         # ② 预生成结构化笔记（Markdown，可导入外部笔记软件）
│   ├── summary/            #    每篇文献的固定模板摘要
│   ├── answers/            #    批量问答记录
│   └── compare/            #    多篇文献横向对比报告
├── data/
│   ├── parsed/             # PDF 解析结果 JSON
│   ├── chunks/             # 多粒度分块 JSON
│   ├── index/              # FAISS 索引、向量缓存、元信息
│   └── papers/             # 网页上传的PDF保存目录（与papers/并列生效）
├── experiments/            # 评估集与消融实验结果
├── logs/                   # 问答与检索调试日志（qa.jsonl / qa.log / retrieve.log / ingest.log）
├── output/                 # 引用网络图、指标对比 CSV 等分析产物
├── requirements.txt
├── .env                    # 密钥配置（不提交 git，自行填写）
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

编辑项目根目录的 `.env`，填写你的智谱AI密钥（[获取地址](https://open.bigmodel.cn)）：

```ini
ZHIPUAI_API_KEY=你的真实密钥
ZHIPUAI_LLM_MODEL=glm-5.3
ZHIPUAI_EMBEDDING_MODEL=embedding-3
EMBEDDING_DIM=2048
```

> 代码只从环境变量读取密钥，**仓库中不存在任何硬编码密钥**；`.env` 已被 `.gitignore` 排除。

### 3. 放入论文并构建索引

```bash
# 将 PDF 放入 papers/ 后：
python -m src.main index        # 增量解析 -> 分块 -> 向量化 -> 建索引
python -m src.main parse --force  # 解析规则升级后强制重解析全部PDF
```

> 解析与向量化均为增量：新放入的 PDF 才会解析，索引中未变化的分块直接复用缓存向量（按内容哈希校验），只有新增/变更分块才调用 embedding API，重复构建不重复扣费。

### 4. 开始提问

```bash
python -m src.main ask "Attention机制的核心思想是什么？"
python -m src.main ask "对比各篇论文的实验指标" --mode hybrid --top-k 8
python -m src.main ask "这篇论文的结论是什么" --source paper1.pdf   # 只在指定PDF内检索
python -m src.main ask "..." --no-rerank                            # 关闭Reranker重排序
python cli_ask.py                                                   # CMD终端交互问答（quit退出）
python -m src.main add        # 增量导入 papers/ 新增PDF（仅新增分块向量化，不全量重建）
```

每条问答自动写入 `logs/` 文件夹：`qa.jsonl`（机器可读）+ `qa.log`（人类可读：参数 + 每个检索片段的检索分/重排分 + 回答全文）；每次检索调试写入 `retrieve.log`，便于排查召回质量。

> **检索策略（自动选择）**：不带 `--source` 提问时，自动启用**按文献均衡召回**——对 `papers/` 里每篇 PDF 独立检索并按篇保底配额（`quota = max(1, top_k // 文献数)`，剩余名额按分数择优），任何一篇都不会被长文档挤出；提示词同时要求**逐篇解答**，输出附【文献覆盖】统计（`paper1.pdf×3块 ...`），未检索到相关片段的文献会单独告警。单独提问某篇请加 `--source`，此时只针对该篇作答。

> **多文档对比建议**：单次检索的 `--top-k` 名额有限（默认 12），联合查询时可调大（建议 `top_k >= 文献数`）。更稳妥的做法是分开查询：每篇单独 `--source` 提问（或直接 `--summary` 生成摘要）后再用 `compare` 对比。

回答会附上 `[1] [2]` 编号对应的引用来源（**原始PDF文件名**、论文标题、章节、页码），并写入 `output/qa_log.jsonl` 便于审计。每个分块的元数据包含 `source`（原始PDF文件名）与 `page`（页码），检索器提供 `metadata_filter` 参数（如 `{"source": "paper1.pdf"}` 或 `{"source": ["a.pdf","b.pdf"]}`，多条件 AND、列表值任一命中），可在指定文献范围内检索作答。开启重排时（默认），检索先取 18 个候选经 `src/reranker.py` 重排再取 top_k 送入大模型；网页问答的「🔎 检索片段明细」可展开查看每块的检索分与重排分。

### 5. 跨论文分析

```bash
python -m src.main analyze
# 生成 output/references.json、citation_network.png、metrics_comparison.csv 等
```

## 文献阅读工作流（预生成笔记优先）

推荐的阅读方式是**先预生成结构化笔记，RAG 仅用于深挖细节**，降低每次实时向量召回不稳定的影响：

```bash
# 1) 为单篇文献预生成固定模板摘要（标题/作者/研究问题/方法/数据/结论/局限）
python -m src.main ask --source paper3.pdf --summary
#    输出保存到 literature_lib/summary/paper3.md，同时控制台打印

# 2) 批量处理：摘要 + 问答一次跑完（编辑 literature_lib/batch_tasks.json）
python batch_run.py --save-markdown
#    任务格式: ["pdf文件名", "问题"]；问题写 "**summary**" 即摘要任务
#    问答 Markdown 可选保存到 literature_lib/answers/

# 3) 多篇横向对比（摘要缺失时自动补生成）
python -m src.main compare --source paper1.pdf --source paper5.pdf
#    对比报告保存到 literature_lib/compare/
```

所有 Markdown 均可直接导入 Obsidian / Typora 等笔记软件复用。

## Web 前端（FastAPI + Gradio）

```bash
python web.py          # 启动并自动打开浏览器 http://127.0.0.1:7860/
python web.py --port 8000 --no-browser   # 自定义端口 / 不自动开浏览器
```

- **同一进程同时提供**：Gradio 前端页面（`/`）与 FastAPI 接口（`/docs` 可查看），业务全部复用 CLI 的底层函数，无重复实现
- **📖 文献库**：展示 papers/ 全部 PDF（标题/页数/是否已解析/是否已有摘要）
- **📄 单篇查看**：选择文献即渲染 `literature_lib/summary/` 对应摘要
- **✨ 生成摘要**：按钮一键调用底层 `generate_summary`，完成后刷新展示
- **💬 问答**：选择单篇（仅答该篇）或全部文献（均衡召回、逐篇解答），展示回答、溯源片段与零召回告警；「🔎 检索片段明细」可展开查看每块的检索分/重排分
- **🔍 检索**：直接查看检索结果与相似度分数（不调用大模型、不消耗额度）
- **⚖️ 对比**：勾选 ≥2 篇文献后两种模式——**填写问题**则围绕问题逐篇检索、生成单篇观点总结并横向对比；**问题留空**则自动执行全文内容对比（研究背景/研究方法/数据集/核心结论/局限性五维度）。证据不足处如实标注，报告存 `literature_lib/compare/`
- REST 接口：`GET /api/papers`、`GET /api/summary/{文件名}`、`POST /api/summary/generate`、`POST /api/ask`、`POST /api/retrieve`、`POST /api/compare`、`POST /api/compare/viewpoints`

## 评估与消融实验

首次运行会自动在 `experiments/eval_set.json` 生成评估集模板，按字段说明填写问题与金标：

```bash
python -m src.evaluator --mode recall --top-k 5     # 检索召回率
python -m src.evaluator --mode ablation             # BM25/稠密/混合/RRF 消融对比
python -m src.evaluator --mode qa --judge keyword   # 问答准确率（关键词判卷，离线）
python -m src.evaluator --mode qa --judge llm       # 问答准确率（GLM-5.3 判卷）
python -m src.evaluator --mode hallucination        # 幻觉率（LLM 逐条事实核查）
python -m src.main eval --mode all --with-qa        # 一键全量评估
```

消融结果保存在 `experiments/ablation_results.json`，终端输出各配置 Recall@k 对比条形图。

## 各模块说明

| 模块 | 职责 | 关键设计 |
| --- | --- | --- |
| `pdf_parser.py` | PDF → 结构化 JSON | 跨页重复行统计去除页眉页脚；编号前缀+关键词正则识别章节；`page_spans` 记录字符偏移到页码的映射 |
| `chunking.py` | 多粒度分块 | 句子级(≈260字)/段落级(≈650字,重叠120)/章节级(≤2600字)；优先用 LlamaIndex SentenceSplitter，缺失时回退内置滑窗 |
| `retrieval.py` | 混合检索 | BM25Okapi + FAISS IndexFlatIP(L2归一化即余弦)；weighted 加权融合或 RRF；向量缓存到 `embeddings.npy` 避免重复调用 |
| `rag_pipeline.py` | 问答与溯源 | System prompt 七条硬约束防编造；低温采样；`[n]` 标记机械校验，无效编号记为幻觉信号；问答双写日志（jsonl + 可读txt） |
| `reranker.py` | 重排序 | 18 候选 → 词覆盖率+二元组匹配+检索分先验重打分 → 取 top_k；纯本地零依赖，可开关 |
| `summarizer.py` | 预生成结构化笔记 | 逐模板字段定向检索证据（6 次定向查询覆盖更全）；固定七字段模板 + 证据不足如实标注；多篇对比报告；问答 Markdown 组装 |
| `paper_analysis.py` | 跨论文分析 | 参考文献双策略切分；标题归一化+difflib 模糊匹配构建引用网络；指标正则抽取（支持中文指标名） |
| `evaluator.py` | 评估与消融 | Recall@k / 准确率(关键词或LLM判卷) / 幻觉率(LLM核查或引用代理) / 4 种检索配置消融 |

## RAG 问答流程

```
papers/*.pdf ──解析──> data/parsed/*.json ──分块──> data/chunks/chunks.json
                                                        │
                              embedding-3 向量化 + BM25 建词表
                                                        ▼
   用户问题 ──> 混合检索(BM25 + FAISS) ──> Top-K 证据块(带 论文/章节/页码)
                                                        │
                              组装受约束 Prompt ──> GLM-5.3 生成
                                                        ▼
   答案 + [n] 引用溯源（溯源到 chunk → 论文-章节-页码） + 幻觉信号校验
```

## 常见问题

- **首次构建索引慢？** 向量化需要调用 embedding-3 API（每篇论文约几十次请求）。向量会缓存到 `data/index/embeddings.npy`，之后重建索引时旧分块自动复用缓存（按内容哈希校验），只向量化新增分块。
- **怎么确认消耗了 API 额度？** 每次问答都写入 `output/qa_log.jsonl`（含答案与引用）；token 用量可登录[智谱开放平台](https://open.bigmodel.cn)在「财务中心/资源用量」中按模型（glm-5.3 / embedding-3）查看调用与消耗记录。
- **扫描版 PDF 解析为空？** pdfplumber 只能抽取有文本层的 PDF；扫描件请先 OCR（如 ABBYY、ocrmypdf）。
- **答案说"无法回答"？** 这是刻意的防编造设计。换一种问法、确认相关论文已在 `papers/` 中并重建索引。
- **章节识别不准？** 当前为启发式规则（标题行 + 关键词），对非标准排版的论文可能有误切；可在 `pdf_parser.py` 的 `SECTION_RULES` 中追加规则，或后续引入基于字号/版面的模型升级。
- **修改了检索配置想重跑？** 删除 `data/index/` 后重新 `python -m src.main index`。

## Roadmap

- [ ] 基于字号/版面信息的章节识别升级
- [ ] 重排序（Rerank）模块接入
- [ ] 扫描版 PDF OCR 支持
- [ ] 简单 Web UI（当前按需求暂缓，优先算法核心）

## 免责说明

仅用于个人学习与科研辅助。模型回答均附引用溯源，但仍请核对原文后再引用；请遵守论文版权与 API 服务条款。
