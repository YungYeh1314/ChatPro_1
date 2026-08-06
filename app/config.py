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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

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
