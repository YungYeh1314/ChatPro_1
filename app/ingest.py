"""数据导入：下载语料（或读本地 JSONL）→ 清洗 → 入库 → 切块 → 向量化 → 存 Chroma。

用法：
  python -m app.ingest download --keywords 计算机 编程 --limit 2000
  python -m app.ingest ingest --file data/sample.jsonl
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
    """字段自适应：不同数据集字段名不一样，按候选顺序取第一个非空值。"""
    for key in keys:
        value = item.get(key)
        if value:
            return str(value)
    return default


def fetch_hf_dataset(keywords: list[str], limit: int, out_path: str) -> None:
    """从 Hugging Face 流式下载中文维基子集，只保留命中关键词的条目。"""
    from datasets import load_dataset

    ds = load_dataset(config.HF_DATASET, split="train", streaming=True)
    pattern = "|".join(keywords)
    saved = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in ds:
            title = _pick(item, ["title", "标题"])
            content = _pick(item, ["content", "text", "body", "正文"])
            if not content:
                continue
            if pattern and not re.search(pattern, title + content):
                continue
            f.write(json.dumps({"title": title, "content": content}, ensure_ascii=False) + "\n")
            saved += 1
            if limit and saved >= limit:
                break
    print(f"完成：已保存 {saved} 条到 {out_path}")


def load_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------- 2. 清洗 ----------

def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)   # 去掉 HTML 标签
    text = re.sub(r"\s+", " ", text)        # 合并多余空白
    return text.strip()


# ---------- 3. 切块 ----------

def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """把长文本切成固定大小的块，相邻块之间留 overlap 字符，避免切断语义。"""
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


# ---------- 4. 向量化 ----------

class Embedder:
    """把文本变成向量。mock 模式生成稳定的伪向量，方便没有 API Key 时测试。"""

    def __init__(self):
        self.mock = config.RAG_MODE == "mock"
        if not self.mock:
            if not config.LLM_API_KEY or config.LLM_API_KEY.startswith("sk-你的"):
                raise SystemExit(
                    "缺少 LLM_API_KEY：请在 .env 中配置，或先使用 RAG_MODE=mock 模式"
                )
            self.client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.mock:
            return [_mock_vector(t) for t in texts]
        resp = self.client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
        return [d.embedding for d in resp.data]


def _mock_vector(text: str, dim: int | None = None) -> list[float]:
    """伪向量：同一段文本永远得到同一个向量（用于流程测试，不是真正的语义向量）。"""
    dim = dim or config.EMBEDDING_DIM
    digest = hashlib.md5(text.encode("utf-8")).digest()
    return [digest[i % len(digest)] / 255.0 for i in range(dim)]


# ---------- 5. 入库主流程 ----------

def ingest_documents(source_file: str | None = None, force: bool = False) -> None:
    database.init_db()

    if source_file is None:
        source_file = str(config.DATA_DIR / "wikipedia_subset.jsonl")
        if not Path(source_file).exists():
            print("未找到本地语料文件。请先下载：python -m app.ingest download")
            print("或指定本地文件：python -m app.ingest ingest --file 你的文件.jsonl")
            sys.exit(1)

    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(
        "docs", metadata={"hnsw:space": "cosine"}
    )
    embedder = Embedder()

    source = Path(source_file).name
    docs_added = chunks_added = 0

    for item in load_jsonl(source_file):
        title = item.get("title", "无标题")
        content = clean_text(item.get("content", ""))
        if len(content) < 20:
            continue  # 太短的正文没有检索价值
        if not force and database.doc_exists(title, source):
            continue  # 已导入过的文档跳过（--force 可强制重导）

        doc_id = database.insert_document(title, source, content)
        chunks = chunk_text(content)

        # 分批向量化（一次 API 调用传多段文本，省请求数）
        ids, metadatas = [], []
        for idx in range(len(chunks)):
            ids.append(f"d{doc_id}-c{idx}")
            metadatas.append({"doc_id": doc_id, "title": title, "chunk_index": idx})
        for start in range(0, len(chunks), 32):
            batch = chunks[start : start + 32]
            vectors = embedder.embed(batch)
            collection.add(
                ids=ids[start : start + 32],
                embeddings=vectors,
                metadatas=metadatas[start : start + 32],
                documents=batch,
            )

        for idx, chunk in enumerate(chunks):
            database.insert_chunk(doc_id, idx, chunk)
        docs_added += 1
        chunks_added += len(chunks)

    print(f"完成：新增 {docs_added} 篇文档、{chunks_added} 个切片")
    print(database.stats())


# ---------- 命令行入口 ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="导入数据到 RAG 知识库")
    sub = parser.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download", help="从 Hugging Face 下载中文维基子集")
    dl.add_argument("--keywords", nargs="+",
                    default=["计算机", "编程", "人工智能", "算法", "数据库", "网络", "Python"],
                    help="主题关键词，只保留标题或正文命中的条目")
    dl.add_argument("--limit", type=int, default=2000, help="最多保存多少条")
    dl.add_argument("--out", type=str, default=None, help="输出 JSONL 路径")

    ing = sub.add_parser("ingest", help="把 JSONL 数据导入知识库")
    ing.add_argument("--file", type=str, default=None, help="本地 JSONL 文件（title/content 字段）")
    ing.add_argument("--force", action="store_true", help="已导入的文档也重新导入")

    args = parser.parse_args()
    if args.cmd == "download":
        out = args.out or str(config.DATA_DIR / "wikipedia_subset.jsonl")
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        fetch_hf_dataset(args.keywords, args.limit, out)
    else:
        ingest_documents(args.file, args.force)


if __name__ == "__main__":
    main()
