"""全局配置：从 .env 读取，未配置时使用默认值。"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ---- LLM API（OpenAI 兼容格式）----
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Embedding 向量化的来源：
#   local = 本地模型（默认，无需额外 API Key，首次使用自动下载模型）
#   api   = OpenAI 兼容的 embedding 接口（可单独配置 Key/地址）
#   mock  = 伪向量，仅供没有 Key 和网络时测试流程
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))

# EMBEDDING_PROVIDER=api 时生效；不填则复用 LLM 的 Key 和地址
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "") or LLM_API_KEY
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "") or LLM_BASE_URL

# mock 模式：不调用真实 LLM，用规则答案跑通全流程（没有 API Key 也能测试）
RAG_MODE = os.getenv("RAG_MODE", "mock")

# ---- 数据与检索 ----
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "rag.db"
CHROMA_DIR = DATA_DIR / "chroma"

TOP_K = int(os.getenv("TOP_K", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Hugging Face 中文维基数据集（字段自适应，也可用本地 JSONL）
HF_DATASET = os.getenv("HF_DATASET", "fjcanyue/wikipedia-zh-cn")
