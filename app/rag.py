"""RAG 核心：检索 → 拼提示词 → 调用 LLM 生成回答。

RAG（Retrieval-Augmented Generation，检索增强生成）的核心思想：
不把大模型当成"百科全书"，而是让它当"阅读理解选手"——
每次提问前，先从自己的知识库里找出最相关的几段资料，
连同问题一起喂给模型，让模型只根据资料回答。

这样做的价值：
1. 资料可以随时增删改，不用重新训练模型；
2. 回答有依据（可以把来源展示给用户），减少"一本正经地胡说八道"。
"""

import chromadb
from openai import OpenAI
from . import config, database
from .ingest import Embedder


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    """把问题向量化，去 Chroma 里找最相似的 top_k 个切片。

    返回值是列表，每个元素是
    {"doc_id", "title", "chunk_index", "content", "score"}。
    """
    top_k = top_k or config.TOP_K
    embedder = Embedder()  # 复用 ingest.py 的向量化器，保证和入库时用同一套向量方式
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    # get_or_create_collection：集合不存在就创建，与入库时保持同一个名字 "docs"
    collection = chroma_client.get_or_create_collection("docs")

    qv = embedder.embed([question])[0]  # 问题也要向量化，变成 512 维向量
    res = collection.query(query_embeddings=[qv], n_results=top_k)
    # query 返回嵌套结构：metadatas / documents / distances 都是
    # [批次][该批次内结果] 两层，这里只有 1 个批次，所以取 [0]。

    results = []
    # 三个列表用 zip 并行遍历：同一位置的元数据、原文、距离属于同一切片
    for meta, doc, dist in zip(
        res["metadatas"][0], res["documents"][0], res["distances"][0]
    ):
        results.append(
            {
                "doc_id": meta.get("doc_id"),
                "title": meta.get("title", "未知来源"),
                "chunk_index": meta.get("chunk_index"),
                "content": doc,
                # Chroma 返回的是"距离"，越小越相似；
                # 这里用 1 - 距离 转成"相似度"，越大越相似，方便前端展示。
                "score": round(1 - float(dist), 4),
            }
        )
    return results


def build_prompt(question: str, contexts: list[dict]) -> str:
    """把检索到的资料拼进提示词，让 LLM 只根据资料回答。

    提示词（prompt）就是喂给大模型的一段文字指令。
    这里用固定模板把"资料"和"问题"组合起来，
    并明确告诉模型：资料里没有的内容要如实说不知道。
    """
    parts = []
    for i, c in enumerate(contexts, 1):  # 编号从 1 开始，方便模型引用资料
        parts.append(f"[资料{i}]《{c['title']}》\n{c['content']}")
    context_block = "\n\n".join(parts)  # 多段资料之间用空行分隔
    return (
        "你是一个基于资料回答问题的助手。只根据提供的资料回答，"
        "资料里没有的内容就如实说明不知道。\n\n"
        f"资料：\n{context_block}\n\n"
        f"问题：{question}\n\n"
        "回答："
    )


def _history_messages(conversation_id: int | None) -> list[dict]:
    """取出某次会话的最近聊天记录，拼成 OpenAI 要求的 messages 格式。

    OpenAI 兼容接口的 messages 是 [{"role": "user"/"assistant", "content": "..."}]。
    把历史消息一起传过去，模型才能记住上下文、进行多轮对话。
    """
    messages = []
    if conversation_id:
        for m in database.get_messages(conversation_id, limit=10):
            messages.append({"role": m["role"], "content": m["content"]})
    return messages


def _mock_answer(question: str, contexts: list[dict]) -> str:
    """mock 模式的"假回答"：不调用任何真实 API，用规则拼一段说明文字。

    用途：没有 API Key 时也能验证"检索 → 前端显示"的整条链路是否通畅。
    它不是一个真正的回答，只是告诉你系统检索到了什么。
    """
    if not contexts:
        return "（mock 模式）知识库里没有相关内容，请先导入数据。"
    names = "、".join(c["title"] for c in contexts[:3])  # 取前 3 个标题做展示
    return (
        f"（mock 模式，未调用真实 LLM）针对“{question}”，检索到 {len(contexts)} 段相关资料，"
        f"其中包含：《{names}》。在 .env 里配置好 API Key 并设置 RAG_MODE=real 后，"
        "这里会变成真正的回答。"
    )


def answer(question: str, conversation_id: int | None = None) -> tuple[str, list[dict]]:
    """非流式回答：一次拿到完整答案，返回 (回答文本, 检索到的来源)。"""
    contexts = retrieve(question)  # 第一步：检索相关切片
    if config.RAG_MODE == "mock":
        return _mock_answer(question, contexts), contexts

    # 校验 Key：为空或者还是 .env.example 的占位符都直接报错，给出可操作的提示
    if not config.LLM_API_KEY or config.LLM_API_KEY.startswith("sk-你的"):
        raise RuntimeError("缺少 LLM_API_KEY：请在 .env 中配置，或使用 RAG_MODE=mock")
    client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    messages = [
        {"role": "system", "content": "你是一个基于资料回答问题的助手。"},  # 系统角色：设定模型身份
        *_history_messages(conversation_id),  # 中间插入历史对话（* 是列表展开语法）
        {"role": "user", "content": build_prompt(question, contexts)},     # 最后是本次问题+资料
    ]
    resp = client.chat.completions.create(model=config.LLM_MODEL, messages=messages)
    return resp.choices[0].message.content, contexts


def stream_answer(question: str, conversation_id: int | None = None):
    """流式回答：逐字返回，用户体验像"打字机"一样。

    返回 (token 生成器, 检索到的来源)。
    生成器每次 yield 一小段文字，chat_api.py 收到后立刻推给前端，
    不用等整段回答全部生成完。
    """
    contexts = retrieve(question)

    if config.RAG_MODE == "mock":
        text = _mock_answer(question, contexts)

        def mock_gen():
            # 按空白切词逐个返回，模拟"逐字输出"的效果
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
    # stream=True 是关键：接口返回的不是完整回答，而是一个"流"，
    # 需要不断迭代 resp 才能拿到陆续到达的碎片。
    resp = client.chat.completions.create(
        model=config.LLM_MODEL, messages=messages, stream=True
    )

    def real_gen():
        for chunk in resp:
            # 每个 chunk 是"增量"：只包含新增的那几个字（delta）
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return real_gen(), contexts
