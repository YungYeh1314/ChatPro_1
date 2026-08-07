"""FastAPI 接口。启动：uvicorn app.chat_api:app --reload

本文件是整个系统的"后端大门"：
- 前端（Streamlit）通过 HTTP 访问这里的接口；
- 接口内部调用 rag.py 完成检索和生成；
- /chat 返回一次性完整回答，/chat/stream 用 SSE 流式逐字返回。
"""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import config, database, rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：控制服务启动和关闭时执行什么。

    yield 之前 = 启动时执行；yield 之后 = 关闭时执行。
    这里启动时初始化数据库（建表）；关闭时暂无全局资源要清理，所以留空。
    这个函数必须是"异步上下文管理器"（async + asynccontextmanager），
    这是 FastAPI 的固定要求，yield 这一行必不可少。
    """
    # 启动时确保 SQLite 数据库和 4 张表存在（CREATE TABLE IF NOT EXISTS，幂等）
    database.init_db()
    yield
    # 关闭时目前没有需要清理的全局资源，留空即可


# 创建 FastAPI 应用，并把生命周期函数注册进去
app = FastAPI(title="ChatPro RAG API", version="0.1.0", lifespan=lifespan)

# CORS 中间件：允许跨域访问。
# 通俗地说，就是"允许其他域名/端口的网页调用我们的接口"。
# 开发阶段放开（*），生产环境应限制为具体的域名。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """请求体模型：pydantic 根据类型自动做数据校验和转换。

    前端 POST 的 JSON 会按这里的字段解析：
    - question 必填，类型必须是字符串；
    - conversation_id 可省略（None），传了会被转成 int。
    校验失败时 FastAPI 会自动返回 422 错误，不需要自己写校验逻辑。
    """
    question: str
    conversation_id: int | None = None


@app.get("/health")
def health() -> dict:
    """健康检查接口：启动后访问 /health 能确认服务活着，并看到当前状态。"""
    # **database.stats() 把字典展开成关键字参数，等价于写
    # {"status": "ok", "mode": config.RAG_MODE, "documents": ..., ...}
    return {"status": "ok", "mode": config.RAG_MODE, **database.stats()}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    """非流式问答接口：一次请求返回完整答案 + 来源 + 会话 id。"""
    if not req.question.strip():  # 去掉首尾空白后为空 = 空问题
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 没有会话 id 就新建一个会话（通常是第一次提问时）
    conv_id = req.conversation_id or database.create_conversation()
    database.add_message(conv_id, "user", req.question)       # 记录用户问题
    answer_text, sources = rag.answer(req.question, conv_id)  # 调用 RAG 核心
    database.add_message(conv_id, "assistant", answer_text)   # 记录助手回答
    return {"answer": answer_text, "sources": sources, "conversation_id": conv_id}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """流式问答接口：用 SSE 逐字推送回答，前端可以实时显示。

    返回类型 StreamingResponse：响应不是一次性 JSON，
    而是一段"事件流"，客户端可以边收边显示。
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    conv_id = req.conversation_id or database.create_conversation()
    database.add_message(conv_id, "user", req.question)
    gen, sources = rag.stream_answer(req.question, conv_id)

    def event_gen():
        """把生成器包装成 SSE 事件序列（本身也是一个生成器）。

        事件协议（前端按 type 字段区分）：
          {"type": "sources", ...} 先发检索到的来源
          {"type": "token", ...}    每段回答文字
          {"type": "done", ...}     回答结束，带上 conversation_id
          {"type": "error", ...}    出错信息
        """
        try:
            # 先发检索到的来源，再流式输出回答
            yield _sse({"type": "sources", "sources": sources})
            full = []
            for token in gen:  # 逐个拿到 LLM 返回的碎片
                full.append(token)
                yield _sse({"type": "token", "content": token})
            answer_text = "".join(full)
            database.add_message(conv_id, "assistant", answer_text)  # 完整回答才存档
            yield _sse({"type": "done", "conversation_id": conv_id})
        except Exception as exc:  # LLM 调用失败时把错误发给前端，而不是断开连接
            yield _sse({"type": "error", "detail": str(exc)})

    # media_type="text/event-stream" 告诉客户端这是 SSE 流
    return StreamingResponse(event_gen(), media_type="text/event-stream")


def _sse(data: dict) -> str:
    """把字典包装成 SSE 格式字符串。

    SSE 规范：每行是 "data: <内容>"，事件之间用空行分隔。
    前端解析时就是按 "data: " 前缀切出后面的 JSON 的。
    """
    return "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"
