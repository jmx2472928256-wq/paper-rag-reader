# -*- coding: utf-8 -*-
"""
项目全局配置模块。

职责
----
1. 加载根目录 .env 环境变量：ZHIPUAI_API_KEY 等敏感信息一律从环境变量
   读取，代码中严禁硬编码任何密钥；
2. 统一管理目录/文件路径，保证各模块读写位置一致；
3. 提供智谱AI能力的统一入口：
   - GLM-5.3      -> 问答推理（rag_pipeline 使用）
   - embedding-3  -> 文本向量化（retrieval 使用）
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# 目录约定
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # 项目根目录

PAPERS_DIR = BASE_DIR / "papers"          # 原始 PDF 存放目录（用户放入）
DATA_DIR = BASE_DIR / "data"              # 中间数据根目录
DATA_PAPERS_DIR = DATA_DIR / "papers"     # 网页上传 PDF 的保存目录（与 papers/ 同级生效）
PARSED_DIR = DATA_DIR / "parsed"          # PDF 解析后的 JSON
CHUNKS_DIR = DATA_DIR / "chunks"          # 分块结果 JSON
INDEX_DIR = DATA_DIR / "index"            # FAISS 索引 / 向量缓存 / 元信息
OUTPUT_DIR = BASE_DIR / "output"          # 分析结果、问答日志等输出
EXPERIMENTS_DIR = BASE_DIR / "experiments"  # 评估集与消融实验结果
# 预生成结构化笔记（文献阅读工具的产出物，Markdown 便于外部笔记软件复用）
LITERATURE_DIR = BASE_DIR / "literature_lib"
SUMMARY_DIR = LITERATURE_DIR / "summary"    # 每篇文献的结构化摘要
ANSWERS_DIR = LITERATURE_DIR / "answers"    # 批量问答 Markdown
COMPARE_DIR = LITERATURE_DIR / "compare"    # 多篇文献横向对比报告

# 各模块共用的关键文件路径
CHUNKS_FILE = CHUNKS_DIR / "chunks.json"      # 全量分块文件
FAISS_FILE = INDEX_DIR / "faiss.index"        # FAISS 索引文件
EMB_FILE = INDEX_DIR / "embeddings.npy"       # 语料向量缓存（避免重复调用 API）
META_FILE = INDEX_DIR / "meta.json"           # 索引元信息（模型名/维度/分块文件路径）
# 问答/检索日志路径见下方 LOG_DIR 配置区（logs/ 文件夹）


def ensure_dirs() -> None:
    """确保所有约定目录存在（幂等，可重复调用）。"""
    for d in (PAPERS_DIR, DATA_PAPERS_DIR, PARSED_DIR, CHUNKS_DIR, INDEX_DIR,
              OUTPUT_DIR, EXPERIMENTS_DIR, SUMMARY_DIR, ANSWERS_DIR,
              COMPARE_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# 加载 .env（文件不存在时静默跳过，全部回退到系统环境变量）
# --------------------------------------------------------------------------
load_dotenv(BASE_DIR / ".env")

# ---- 智谱AI密钥：仅从环境变量读取，代码中不存在任何硬编码密钥 ----
ZHIPUAI_API_KEY: str = os.getenv("ZHIPUAI_API_KEY", "").strip()

# ---- top_k 边界保护（任务⑤）----
TOP_K_DEFAULT = 12                # 默认送入大模型的分块数
TOP_K_MIN = 1
TOP_K_MAX = 50


def clamp_top_k(value, default: int = TOP_K_DEFAULT) -> int:
    """
    top_k 数值边界保护：非法输入回退默认值，超界钳制到
    [TOP_K_MIN, TOP_K_MAX]，防止非法数字引发崩溃或提示词爆炸。
    """
    if value is None:
        return default
    try:
        v = int(value)
    except (TypeError, ValueError):
        print(f"[config] top_k={value!r} 不是合法整数，使用默认值 {default}")
        return default
    if v < TOP_K_MIN:
        print(f"[config] top_k={v} 低于下界 {TOP_K_MIN}，已钳制")
        v = TOP_K_MIN
    elif v > TOP_K_MAX:
        print(f"[config] top_k={v} 超过上界 {TOP_K_MAX}，已钳制")
        v = TOP_K_MAX
    return v


def _env_int(name: str, default: int, lo: int = None, hi: int = None) -> int:
    """安全读取整型环境变量：非法值回退默认并告警，可选范围钳制（任务⑤）。"""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        print(f"[config] 环境变量 {name}={raw!r} 不是合法整数，使用默认值 {default}")
        return default
    if lo is not None and v < lo:
        print(f"[config] {name}={v} 低于下界 {lo}，已钳制")
        v = lo
    if hi is not None and v > hi:
        print(f"[config] {name}={v} 超过上界 {hi}，已钳制")
        v = hi
    return v


def _env_float(name: str, default: float, lo: float = None, hi: float = None) -> float:
    """安全读取浮点环境变量（非法值回退默认并告警，可选范围钳制）。"""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
    except ValueError:
        print(f"[config] 环境变量 {name}={raw!r} 不是合法数字，使用默认值 {default}")
        return default
    if lo is not None and v < lo:
        v = lo
    if hi is not None and v > hi:
        v = hi
    return v


def _env_bool(name: str, default: bool) -> bool:
    """安全读取布尔环境变量（1/true/yes/on 为真，0/false/no/off 为假）。"""
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    print(f"[config] 环境变量 {name}={raw!r} 无法识别为布尔值，使用默认值 {default}")
    return default

# ---- 模型约定（可通过 .env 覆盖；非法取值自动回退默认并告警） ----
LLM_MODEL: str = os.getenv("ZHIPUAI_LLM_MODEL", "glm-5.3").strip() or "glm-5.3"
EMBEDDING_MODEL: str = os.getenv("ZHIPUAI_EMBEDDING_MODEL", "embedding-3").strip() or "embedding-3"
# Step Plan 配置（OpenAI 兼容接口；留空则自动回退到智谱 GLM / embedding）
STEP_API_KEY: str = os.getenv("STEP_API_KEY", "").strip()
STEP_BASE_URL: str = os.getenv("STEP_BASE_URL", "https://api.stepfun.com/step_plan/v1").strip()
STEP_LLM_MODEL: str = os.getenv("STEP_LLM_MODEL", "step-3.7-flash").strip() or "step-3.7-flash"
# StepPlan Embedding 配置（当 ZHIPUAI_API_KEY 为空/无效时，回退到 StepPlan）
STEP_EMBEDDING_BASE_URL: str = os.getenv("STEP_EMBEDDING_BASE_URL", "https://api.stepfun.com/v1").strip()
STEP_EMBEDDING_MODEL: str = os.getenv("STEP_EMBEDDING_MODEL", "step-embedding").strip() or "step-embedding"
# 默认 embedding 维度：本地 BGE-small-zh-v1.5 固定 512 维
_EMBEDDING_DIM_RAW = _env_int("EMBEDDING_DIM", 0, lo=64, hi=8192)
if _EMBEDDING_DIM_RAW in (256, 512, 1024, 2048):
    EMBEDDING_DIM: int = _EMBEDDING_DIM_RAW
else:
    EMBEDDING_DIM = 512  # bge-small-zh-v1.5 输出维度

# ---- 日志（每次问答/检索调试自动写入 logs/ 文件夹） ----
LOG_DIR = BASE_DIR / "logs"
QA_LOG_FILE = LOG_DIR / "qa.jsonl"        # 机器可读问答日志（全量字段）
QA_TEXT_LOG = LOG_DIR / "qa.log"          # 人类可读问答日志（问题/片段/回答/参数）
RETRIEVE_LOG = LOG_DIR / "retrieve.log"   # 检索调试日志（query/参数/每块分数与文本）
INGEST_LOG = LOG_DIR / "ingest.log"       # 网页上传日志（成功/重复/解析失败明细）

# ---- 其他默认超参数 ----
EMBED_BATCH_SIZE: int = _env_int("EMBED_BATCH_SIZE", 16, 1, 64)   # 每次向量化请求的文本条数
EMBED_MAX_CHARS: int = _env_int("EMBED_MAX_CHARS", 6000, 500, 8000)  # 单条送入 embedding 的最大字符数
LLM_TEMPERATURE: float = 0.1      # 问答温度：低温抑制发挥，降低编造概率
LLM_MAX_RETRIES: int = 3          # API 调用失败重试次数

# ---- 分块配置（中期迭代） ----
# 分块模式: char = 字符滑窗（默认，行为与历史版本一致）；semantic = 语义分块
# （TextTiling 风格，在相邻窗口相似度谷底切分，零第三方依赖）。可用
# 环境变量 CHUNK_MODE 或 `python -m src.chunking --mode semantic` 切换。
_CHUNK_MODE_RAW = os.getenv("CHUNK_MODE", "char").strip().lower()
if _CHUNK_MODE_RAW in ("char", "semantic"):
    CHUNK_MODE: str = _CHUNK_MODE_RAW
else:
    print(f"[config] CHUNK_MODE={_CHUNK_MODE_RAW!r} 不合法(仅支持 char/semantic)，回退 char")
    CHUNK_MODE: str = "char"
# 块间重叠比例：overlap = chunk_size * CHUNK_OVERLAP_RATIO（默认 15%，钳制 [0, 50%]）
CHUNK_OVERLAP_RATIO: float = _env_float("CHUNK_OVERLAP_RATIO", 0.15, 0.0, 0.5)

# ---- Reranker 重排序（中期迭代） ----
# 总开关：检索先取 RERANK_CANDIDATES 个候选，重排后再取 top_k 送入大模型
RERANK_ENABLED: bool = _env_bool("RERANK_ENABLED", True)
RERANK_CANDIDATES: int = _env_int("RERANK_CANDIDATES", 18, 1, 200)


_client = None  # 智谱AI客户端单例缓存
_step_client = None  # StepPlan OpenAI 客户端单例缓存


def get_zhipu_client():
    """
    获取智谱AI客户端（懒加载单例）。

    Returns:
        zhipuai.ZhipuAI 客户端实例。

    Raises:
        RuntimeError: 未配置 ZHIPUAI_API_KEY 时抛出，并给出配置指引。
    """
    global _client
    if _client is None:
        if not ZHIPUAI_API_KEY:
            raise RuntimeError(
                "未检测到 ZHIPUAI_API_KEY！请在项目根目录的 .env 文件中填写：\n"
                "    ZHIPUAI_API_KEY=你的真实密钥\n"
                "密钥获取地址: https://open.bigmodel.cn"
            )
        # 延迟导入：只有真正需要调用 API 时才要求安装 zhipuai
        try:
            from zhipuai import ZhipuAI
        except ImportError as e:
            raise RuntimeError("请先安装智谱AI SDK：pip install zhipuai") from e
        _client = ZhipuAI(api_key=ZHIPUAI_API_KEY)
    return _client


def get_step_client():
    """
    获取 StepPlan 客户端（OpenAI 兼容接口，懒加载单例）。

    Returns:
        openai.OpenAI 客户端实例（base_url 指向 STEP_BASE_URL）。

    Raises:
        RuntimeError: 未配置 STEP_API_KEY 时抛出。
    """
    global _step_client
    if _step_client is None:
        if not STEP_API_KEY:
            raise RuntimeError(
                "未检测到 STEP_API_KEY！请在 .env 中填写 StepPlan 密钥，"
                "或留空 STEP_API_KEY 以回退到智谱 GLM。"
            )
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("请先安装OpenAI SDK：pip install openai") from e
        _step_client = OpenAI(api_key=STEP_API_KEY, base_url=STEP_BASE_URL)
    return _step_client


def chat(messages, temperature: float = LLM_TEMPERATURE,
         max_retries: int = LLM_MAX_RETRIES, prefer_step: bool | None = None) -> str:
    """
    调用问答推理大模型（Step Plan 优先，未配置时自动回退智谱 GLM）。

    后端选择优先级：
    1. ``prefer_step=True``   -> 强制 Step Plan（OpenAI 兼容接口）
    2. ``prefer_step=False``  -> 强制智谱 GLM
    3. ``prefer_step=None``（默认）-> 检测到 STEP_API_KEY 时用 Step Plan，
       否则回退智谱 GLM。

    Args:
        messages: OpenAI 风格消息列表。
        temperature: 采样温度。
        max_retries: 重试次数。
        prefer_step: 显式指定后端。

    Returns:
        模型回答文本。
    """
    if prefer_step is None:
        prefer_step = bool(STEP_API_KEY)

    if prefer_step:
        client = get_step_client()
        model = STEP_LLM_MODEL
        backend = "StepPlan"
    else:
        client = get_zhipu_client()
        model = LLM_MODEL
        backend = "ZhipuGLM"

    last_err: Exception = RuntimeError("chat: 未知错误")
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            msg = resp.choices[0].message
            content = (getattr(msg, "content", None) or "").strip()
            if not content:
                content = (getattr(msg, "reasoning_content", None) or "").strip()
            if content:
                return content
            last_err = RuntimeError("chat: 模型返回了空内容")
        except Exception as e:
            last_err = e
            msg = str(e)
            if "429" in msg or "限流" in msg or "rate" in msg.lower() or "并发" in msg:
                print(f"[config] 疑似触发API限流（第{attempt}/{max_retries}次），"
                      f"退避{2.0 * attempt:.0f}s后重试...")
            else:
                print(f"[config] LLM 调用失败（第{attempt}/{max_retries}次）: {e}")
            time.sleep(2.0 * attempt)
    if prefer_step:
        raise RuntimeError(
            f"LLM 调用失败（已重试{max_retries}次）: {last_err}\n"
            f"可能原因与建议: ①API限流/并发超限——稍等几秒重试；②额度不足——"
            f"查看 Step Plan 账户余额；③密钥无效——检查 .env 的 STEP_API_KEY；"
            f"④BASE_URL 错误——当前: {STEP_BASE_URL}；⑤网络异常——检查本机网络。"
        )
    raise RuntimeError(
        f"LLM 调用失败（已重试{max_retries}次）: {last_err}\n"
        f"可能原因与建议: ①API限流/并发超限——稍等几秒重试；②额度不足——"
        f"前往 open.bigmodel.cn 查看余额；③密钥无效——检查 .env 中 "
        f"ZHIPUAI_API_KEY；④网络异常——检查本机网络。")


def embed_texts(texts, batch_size: int = EMBED_BATCH_SIZE,
                max_retries: int = LLM_MAX_RETRIES):
    """
    批量文本向量化（本地 BGE-small-zh-v1.5，离线可用）。

    加载优先级：
    1. 本地目录 models/bge-small-zh-v1.5 存在 -> 直接加载，不联网
    2. 否则通过 HF_ENDPOINT 镜像下载到本地缓存后再加载
    """
    if not texts:
        return []
    return _embed_local_bge(texts, batch_size)


def _model_cache_dir() -> Path:
    """SentenceTransformer 缓存目录（优先项目内 models/，避免污染用户目录）"""
    return BASE_DIR / "models"


def _ensure_local_bge() -> "SentenceTransformer":
    """
    确保本地 BGE-small-zh-v1.5 可用：
    - 本地目录 models/bge-small-zh-v1.5 存在且完整 -> 直接加载
    - 否则走 HF_ENDPOINT 镜像下载到本地缓存 -> 再加载
    """
    from sentence_transformers import SentenceTransformer

    local_dir = _model_cache_dir() / "bge-small-zh-v1.5"
    # 判断本地是否已有完整模型文件（config.json + pytorch_model.bin）
    has_local = (local_dir / "config.json").exists() and (
        (local_dir / "pytorch_model.bin").exists()
        or (local_dir / "model.safetensors").exists()
    )
    if has_local:
        print(f"[config] 本地BGE模型已存在，直接加载: {local_dir}")
        return SentenceTransformer(str(local_dir), device="cpu")

    # 本地不存在，走 HF 镜像下载到缓存目录
    if not STEP_API_KEY:
        # 即使不走 StepPlan LLM，embedding 也要求有网络下载一次模型
        print("[config] 本地BGE模型不存在，将尝试从 HF 镜像下载（首次需要联网）")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cpu")
    # 下载后保存到项目 models/ 目录，后续离线可用
    local_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(local_dir))
    print(f"[config] BGE模型已下载到本地: {local_dir}")
    return model


_bge_model = None  # 本地BGE单例


def _embed_local_bge(texts, batch_size, max_retries=1):
    """本地 BGE-small-zh-v1.5 推理（离线可用，不调用任何线上API）。"""
    global _bge_model
    if _bge_model is None:
        _bge_model = _ensure_local_bge()
    # SentenceTransformer.encode 是同步推理，不会触发网络请求
    vectors = _bge_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vectors]


def setup_console() -> None:
    """
    Windows 控制台默认 GBK 编码，打印中文/特殊符号可能报 UnicodeEncodeError。
    在入口处调用本函数，尽量将标准流切换为 UTF-8（失败则静默忽略）。
    """
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # 启动提示：打印 LLM 与 Embedding 配置
    try:
        llm_backend = "StepPlan(step-router-v1)" if STEP_API_KEY else "ZhipuGLM(glm-5.3)"
        embed_backend = "本地BGE-small-zh-v1.5(512维)"
        print(f"[启动配置] LLM={llm_backend} | Embedding={embed_backend}")
    except Exception:
        pass
