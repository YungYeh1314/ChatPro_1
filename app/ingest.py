"""数据导入：下载语料（或读本地 JSONL）→ 清洗 → 入库 → 切块 → 向量化 → 存 Chroma。

这是 RAG 系统的"离线建库"环节，只在导入数据时运行一次。
运行方式（两种）：
  python -m app.ingest download --keywords 计算机 编程 --limit 2000
  python -m app.ingest ingest --file data/sample.jsonl

注意必须用 `python -m app.ingest`（模块方式）运行：
这样 Python 才知道 app 是一个包，文件内部的 `from . import config`
（相对导入）才能正常工作。直接 python app/ingest.py 会因缺少包上下文而报错。
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import chromadb
from openai import OpenAI

from . import config, database


# ---------- 1. 数据获取 ----------

def _pick(item: dict, keys: list[str], default: str = "") -> str:
    """字段自适应：不同数据集字段名不一样，按候选顺序取第一个非空值。

    例如有的数据集正文字段叫 "content"，有的叫 "text" 或 "body"，
    传入 ["content", "text", "body", "正文"] 就能兼容多种格式。
    item.get(key) 取不到时返回 None（假值），就继续尝试下一个 key。
    """
    for key in keys:
        value = item.get(key)
        if value:
            return str(value)
    return default


def fetch_hf_dataset(keywords: list[str], limit: int, out_path: str) -> None:
    """从 Hugging Face 流式下载中文维基子集，只保留命中关键词的条目。

    - streaming=True：不把整个数据集一次性加载进内存，而是一条条读取，省内存。
    - pattern = "|".join(keywords) 把关键词拼成正则 "计算机|编程|..."，
      re.search 检查标题或正文里是否命中任意一个关键词。
    - 每命中一条就写一行 JSON 到 JSONL 文件（每行是一个完整的 JSON 对象）。
    """
    from datasets import load_dataset
    # 延迟导入：只有真正执行 download 时才加载 datasets 库，
    # 平时 import 本模块不引入这个大依赖，程序启动更快。

    ds = load_dataset(config.HF_DATASET, split="train", streaming=True)
    pattern = "|".join(keywords)
    saved = 0
    with open(out_path, "w", encoding="utf-8") as f:
        # encoding="utf-8" 必须显式指定，否则 Windows 上默认按 GBK 写会乱码。
        for item in ds:
            title = _pick(item, ["title", "标题"])
            content = _pick(item, ["content", "text", "body", "正文"])
            if not content:
                continue  # 没有正文的条目没有检索价值，跳过
            if pattern and not re.search(pattern, title + content):
                continue  # 关键词没命中，跳过
            f.write(json.dumps({"title": title, "content": content}, ensure_ascii=False) + "\n")
            # ensure_ascii=False：不要把中文转成 \uXXXX 转义，文件里直接存中文。
            saved += 1
            if limit and saved >= limit:
                break  # 达到数量上限就停
    print(f"完成：已保存 {saved} 条到 {out_path}")


def load_jsonl(path: str):
    """逐行读取 JSONL 文件，每行 yield 一个字典。

    yield 让这个函数变成"生成器"：调用时不立刻执行，
    而是每次迭代时执行到下一个 yield，返回一行数据。
    好处：无论文件多大，内存里始终只放一行。
    """
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # 去掉行首行尾空白
            if line:             # 跳过空行
                yield json.loads(line)


# ---------- 2. 清洗 ----------

def clean_text(text: str) -> str:
    """清洗文本：去掉 HTML 标签、合并多余空白。

    真实语料里常混有 <p>、<div> 之类的标签，检索前先清掉，
    避免把标签本身当成内容存进向量库。
    """
    text = re.sub(r"<[^>]+>", " ", text)   # 正则：< 开头，> 结尾的一段 → 替换成空格
    text = re.sub(r"\s+", " ", text)        # 多个空白（空格/换行/tab）合并成一个空格
    return text.strip()                     # 去掉首尾空白


# ---------- 3. 切块 ----------

def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """把长文本切成固定大小的块，相邻块之间留 overlap 字符，避免切断语义。

    为什么要切片：
    - 大模型上下文长度有限，不能把整本书都塞进去；
    - 检索需要"小而聚焦"的片段，整篇内容混在一起反而搜不准。
    为什么要 overlap：一句话可能正好被边界切开，重叠一段能让下一块"接上话"。

    算法示意（size=10, overlap=3）：
      第一块：text[0:10]
      第二块：text[7:17]   <- 起点 = 上次结束 10 - 重叠 3
    """
    size = size or config.CHUNK_SIZE       # 传了就用参数，没传用配置里的默认值
    overlap = overlap or config.CHUNK_OVERLAP
    if len(text) <= size:
        return [text]                      # 文本很短就不用切
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap              # 下一块从"上次结尾往前挪 overlap"开始
    return chunks


# ---------- 4. 向量化 ----------

class Embedder:
    """把文本变成向量，支持三种来源：
    - local：本地模型（fastembed + bge-small-zh，中文效果好，无需额外 API Key）
    - api：OpenAI 兼容的 embedding 接口（可单独配置 EMBEDDING_BASE_URL / KEY）
    - mock：伪向量，用于没有网络和 Key 时测试流程

    什么是向量：把一段文字变成一串数字（这里是 512 个小数）。
    语义相近的文字，它们的数字向量在空间里的距离也更近——
    这就是向量库能"按语义找相似内容"的原理。
    """

    def __init__(self):
        # 选择向量化方式：
        # 显式配置了 EMBEDDING_PROVIDER 就用它；
        # 没配置时，mock 模式用伪向量，real 模式用本地模型。
        self.provider = (
            config.EMBEDDING_PROVIDER
            or ("mock" if config.RAG_MODE == "mock" else "local")
        )
        if self.provider == "api":
            # 用 API 做向量化必须有 Key，且不能是 .env.example 里的占位符
            if not config.EMBEDDING_API_KEY or config.EMBEDDING_API_KEY.startswith("sk-你的"):
                raise SystemExit(
                    "EMBEDDING_PROVIDER=api 但缺少 EMBEDDING_API_KEY，请在 .env 中配置"
                )
            # OpenAI() 客户端是一个"兼容层"：只要接口格式兼容 OpenAI 的都能用
            # （DeepSeek、GLM 等），只需换 base_url。
            self.client = OpenAI(
                api_key=config.EMBEDDING_API_KEY,
                base_url=config.EMBEDDING_BASE_URL,
            )
        elif self.provider == "local":
            self._init_local()

    def _init_local(self) -> None:
        """初始化本地 embedding 模型（首次使用会自动下载，约 100MB）。"""
        try:
            from fastembed import TextEmbedding
        except ImportError:
            # 依赖没装时给出明确提示，而不是抛一个晦涩的异常
            raise SystemExit("未安装 fastembed：请在虚拟环境执行 pip install fastembed")
        self._model = TextEmbedding(model_name=config.EMBEDDING_MODEL)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """把一个文本列表变成向量列表（一次处理一批，省请求数）。"""
        if self.provider == "mock":
            return [_mock_vector(t) for t in texts]
        if self.provider == "api":
            resp = self.client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
            return [d.embedding for d in resp.data]
        # local：fastembed 返回 numpy 数组，转成 list 才能被 Chroma 接受
        return [v.tolist() for v in self._model.embed(texts)]


def _mock_vector(text: str, dim: int | None = None) -> list[float]:
    """伪向量：同一段文本永远得到同一个向量（用于流程测试，不是真正的语义向量）。

    原理：对文本取 MD5 哈希（16 字节的确定值），再把字节按 dim 循环展开成向量。
    同一文本 → 同一哈希 → 同一向量；不同文本 → 大概率不同的向量。
    所以 mock 模式也能"检索"，但检索结果只是巧合，不代表语义相关。
    """
    dim = dim or config.EMBEDDING_DIM
    digest = hashlib.md5(text.encode("utf-8")).digest()
    return [digest[i % len(digest)] / 255.0 for i in range(dim)]  # 字节/255 归一化到 0~1


# ---------- 5. 入库主流程 ----------

def ingest_documents(source_file: str | None = None, force: bool = False) -> None:
    """建库主流程：读 JSONL → 清洗 → 原文入库 SQLite → 切块 → 向量化 → 存 Chroma。

    force=True 时，已导入过的文档也会重新导入（会重复入库，一般不推荐）。
    """
    database.init_db()  # 确保 SQLite 的 4 张表存在

    if source_file is None:
        # 不指定文件时，默认找 data/wikipedia_subset.jsonl
        source_file = str(config.DATA_DIR / "wikipedia_subset.jsonl")
        if not Path(source_file).exists():
            print("未找到本地语料文件。请先下载：python -m app.ingest download")
            print("或指定本地文件：python -m app.ingest ingest --file 你的文件.jsonl")
            sys.exit(1)  # 退出程序并返回非 0 状态码，表示异常结束

    # 打开（或创建）Chroma 向量库
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    # 集合（collection）类似数据库里的"表"；hnsw:space=cosine 表示用余弦相似度检索。
    collection = chroma_client.get_or_create_collection(
        "docs", metadata={"hnsw:space": "cosine"}
    )
    embedder = Embedder()  # 创建向量化器（local 模式这一步会加载模型）

    source = Path(source_file).name  # 用文件名当作"来源"标记，如 sample_wikipedia.jsonl
    docs_added = chunks_added = 0

    for item in load_jsonl(source_file):
        title = item.get("title", "无标题")
        content = clean_text(item.get("content", ""))
        if len(content) < 20:
            continue  # 太短的正文没有检索价值
        if not force and database.doc_exists(title, source):
            continue  # 已导入过的文档跳过（--force 可强制重导）

        # 1) 原文入库 SQLite，拿到自增 doc_id
        doc_id = database.insert_document(title, source, content)
        # 2) 把长文切成若干块（默认每块 500 字符、重叠 100）
        chunks = chunk_text(content)

        # 3) 向量化并写入 Chroma
        # 每块生成一个唯一 id（如 d1-c0），并把 doc_id/title 等信息放进元数据，
        # 这样检索命中后能反查到 SQLite 里的原文和标题。
        ids, metadatas = [], []
        for idx in range(len(chunks)):
            ids.append(f"d{doc_id}-c{idx}")
            metadatas.append({"doc_id": doc_id, "title": title, "chunk_index": idx})
        # 分批向量化：每批最多 32 段，一次 API 调用传多段文本，省请求数。
        for start in range(0, len(chunks), 32):
            batch = chunks[start : start + 32]
            vectors = embedder.embed(batch)
            collection.add(
                ids=ids[start : start + 32],
                embeddings=vectors,
                metadatas=metadatas[start : start + 32],
                documents=batch,  # Chroma 里同时存原文，检索时可直接返回
            )

        # 4) 切片也写一份到 SQLite（chunks 表），与 Chroma 通过 doc_id 对应
        for idx, chunk in enumerate(chunks):
            database.insert_chunk(doc_id, idx, chunk)
        docs_added += 1
        chunks_added += len(chunks)

    print(f"完成：新增 {docs_added} 篇文档、{chunks_added} 个切片")
    print(database.stats())


# ---------- 命令行入口 ----------

def main() -> None:
    """命令行入口：解析 python -m app.ingest download/ingest 后面的参数。"""
    parser = argparse.ArgumentParser(description="导入数据到 RAG 知识库")
    # 子命令机制：本文件支持 download 和 ingest 两个子命令
    sub = parser.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download", help="从 Hugging Face 下载中文维基子集")
    dl.add_argument("--keywords", nargs="+",  # nargs="+"：后面可以跟一个或多个词
                    default=["计算机", "编程", "人工智能", "算法", "数据库", "网络", "Python"],
                    help="主题关键词，只保留标题或正文命中的条目")
    dl.add_argument("--limit", type=int, default=2000, help="最多保存多少条")
    dl.add_argument("--out", type=str, default=None, help="输出 JSONL 路径")

    ing = sub.add_parser("ingest", help="把 JSONL 数据导入知识库")
    ing.add_argument("--file", type=str, default=None, help="本地 JSONL 文件（title/content 字段）")
    ing.add_argument("--force", action="store_true", help="已导入的文档也重新导入")
    # action="store_true"：命令行里出现 --force 就是 True，不出现就是 False

    args = parser.parse_args()
    if args.cmd == "download":
        out = args.out or str(config.DATA_DIR / "wikipedia_subset.jsonl")
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        fetch_hf_dataset(args.keywords, args.limit, out)
    else:
        ingest_documents(args.file, args.force)


# 同样的"入口"惯用法：只有 python -m app.ingest 直接运行时才执行 main()，
# 被 rag.py 等文件 import 时不会触发。
if __name__ == "__main__":
    main()
