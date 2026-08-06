"""RAG 核心：检索 → 拼提示词 → 调用 LLM 生成回答。"""

import chromadb
from openai import OpenAI

from . import config, database
from .ingest import Embedder


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    """把问题向量化，去 Chroma 里找最相似的 top_k 个切片。"""
    top_k = top_k or config.TOP_K
    embedder = Embedder()
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = chroma_client.get_or_create_collection("docs")

    qv = embedder.embed([question])[0]
    res = collection.query(query_embeddings=[qv], n_results=top_k)

    results = []
    for meta, doc, dist in zip(
        res["metadatas"][0], res["documents"][0], res["distances"][0]
    ):
        results.append(
            {
                "doc_id": meta.get("doc_id"),
                "title": meta.get("title", "未知来源"),
                "chunk_index": meta.get("chunk_index"),
                "content": doc,
                "score": round(1 - float(dist), 4),  # cosine 距离转相似度
            }
        )
    return results


def build_prompt(question: str, contexts: list[dict]) -> str:
    """把检索到的资料拼进提示词，让 LLM 只根据资料回答。"""
    parts = []
    for i, c in enumerate(contexts, 1):
        parts.append(f"[资料{i}]《{c['title']}》\n{c['content']}")
    context_block = "\n\n".join(parts)
    return (
        "你是一个基于资料回答问题的助手。只根据提供的资料回答，"
        "资料里没有的内容就如实说明不知道。\n\n"
        f"资料：\n{context_block}\n\n"
        f"问题：{question}\n\n"
        "回答："
    )


def _history_messages(conversation_id: int | None) -> list[dict]:
    messages = []
    if conversation_id:
        for m in database.get_messages(conversation_id, limit=10):
            messages.append({"role": m["role"], "content": m["content"]})
    return messages


def _mock_answer(question: str, contexts: list[dict]) -> str:
    if not contexts:
        return "（mock 模式）知识库里没有相关内容，请先导入数据。"
    names = "、".join(c["title"] for c in contexts[:3])
    return (
        f"（mock 模式，未调用真实 LLM）针对“{question}”，检索到 {len(contexts)} 段相关资料，"
        f"其中包含：《{names}》。在 .env 里配置好 API Key 并设置 RAG_MODE=real 后，"
        "这里会变成真正的回答。"
    )


def answer(question: str, conversation_id: int | None = None) -> tuple[str, list[dict]]:
    """非流式：返回 (回答文本, 检索到的来源)。"""
    contexts = retrieve(question)
    if config.RAG_MODE == "mock":
        return _mock_answer(question, contexts), contexts

    if not config.LLM_API_KEY or config.LLM_API_KEY.startswith("sk-你的"):
        raise RuntimeError("缺少 LLM_API_KEY：请在 .env 中配置，或使用 RAG_MODE=mock")
    client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    messages = [
        {"role": "system", "content": "你是一个基于资料回答问题的助手。"},
        *_history_messages(conversation_id),
        {"role": "user", "content": build_prompt(question, contexts)},
    ]
    resp = client.chat.completions.create(model=config.LLM_MODEL, messages=messages)
    return resp.choices[0].message.content, contexts


def stream_answer(question: str, conversation_id: int | None = None):
    """流式：返回 (token 生成器, 检索到的来源)。"""
    contexts = retrieve(question)

    if config.RAG_MODE == "mock":
        text = _mock_answer(question, contexts)

        def mock_gen():
            for token in text.split():
                yield token + " "

        return mock_gen(), contexts

    if not config.LLM_API_KEY or config.LLM_API_KEY.startswith("sk-你的"):
        raise RuntimeError("缺少 LLM_API_KEY：请在 .env 中配置，或使用 RAG_MODE=mock")
    client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    messages = [
        {"role": "system", "content": "你是一个基于资料回答问题的助手。"},
        *_history_messages(conversation_id),
        {"role": "user", "content": build_prompt(question, contexts)},
    ]
    resp = client.chat.completions.create(
        model=config.LLM_MODEL, messages=messages, stream=True
    )

    def real_gen():
        for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return real_gen(), contexts
