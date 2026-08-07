"""全局配置：从 .env 读取，未配置时使用默认值。

设计思路：把所有"可能会变"的东西（API Key、模型名、切块大小、检索条数……）
集中放在这里，而不是散落在各个文件里。
以后改配置只动 .env 文件，完全不用改代码。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# BASE_DIR = 项目根目录（ChatPro_1/）
# __file__ 是当前文件（config.py）的完整路径；
# .resolve() 把相对路径转成绝对路径；.parent 往上一级取目录。
# config.py 位于 ChatPro_1/app/ 里，所以 parent.parent 才能回到 ChatPro_1/。
BASE_DIR = Path(__file__).resolve().parent.parent
# 读取项目根目录下的 .env，把里面的 KEY=VALUE 全部写进环境变量。
# load_dotenv 默认不会覆盖系统里已有的环境变量。
load_dotenv(BASE_DIR / ".env")

# ---- LLM API（OpenAI 兼容格式）----
# os.getenv("名字", "默认值")：从环境变量取值，取不到就用默认值。
# 这样即使 .env 里少写某个配置，程序也不会崩，只是用默认值。
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Embedding 向量化的来源：
#   local = 本地模型（默认，无需额外 API Key，首次使用自动下载模型）
#   api   = OpenAI 兼容的 embedding 接口（可单独配置 Key/地址）
#   mock  = 伪向量，仅供没有 Key 和网络时测试流程
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
# 向量维度必须和 embedding 模型的输出维度一致（bge-small-zh-v1.5 输出 512 维）。
# 更换 embedding 模型时，要同步改维度，并清空 data/chroma/ 后重新导入数据。
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))
# 注意：环境变量读出来一定是字符串，所以这里用 int() 转成整数再使用。

# EMBEDDING_PROVIDER=api 时生效；不填则复用 LLM 的 Key 和地址
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "") or LLM_API_KEY
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "") or LLM_BASE_URL
# 上面用 or 做"回退"：第一项为空字符串（假值）时，自动取第二项的值。

# mock 模式：不调用真实 LLM，用规则答案跑通全流程（没有 API Key 也能测试）
RAG_MODE = os.getenv("RAG_MODE", "mock")

# ---- 数据与检索 ----
DATA_DIR = BASE_DIR / "data"         # 所有运行时数据都放在这里
DB_PATH = DATA_DIR / "rag.db"        # SQLite 数据库文件
CHROMA_DIR = DATA_DIR / "chroma"     # Chroma 向量库目录

TOP_K = int(os.getenv("TOP_K", "4"))                 # 每次检索返回最相似的几段
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))     # 每段切片的最大字符数
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))  # 相邻切片重叠的字符数

# Hugging Face 中文维基数据集（字段自适应，也可用本地 JSONL）
HF_DATASET = os.getenv("HF_DATASET", "fjcanyue/wikipedia-zh-cn")
