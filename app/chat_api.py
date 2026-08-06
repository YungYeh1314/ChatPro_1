"""FastAPI 接口。启动：uvicorn app.chat_api:app --reload"""

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import config, database, rag

app = FastAPI(title="ChatPro RAG API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    conversation_id: int | None = None


@app.on_event("startup")
def startup() -> None:
    database.init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mode": config.RAG_MODE, **database.stats()}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    conv_id = req.conversation_id or database.create_conversation()
    database.add_message(conv_id, "user", req.question)
    answer_text, sources = rag.answer(req.question, conv_id)
    database.add_message(conv_id, "assistant", answer_text)
    return {"answer": answer_text, "sources": sources, "conversation_id": conv_id}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    conv_id = req.conversation_id or database.create_conversation()
    database.add_message(conv_id, "user", req.question)
    gen, sources = rag.stream_answer(req.question, conv_id)

    def event_gen():
        try:
            # 先发检索到的来源，再流式输出回答
            yield _sse({"type": "sources", "sources": sources})
            full = []
            for token in gen:
                full.append(token)
                yield _sse({"type": "token", "content": token})
            answer_text = "".join(full)
            database.add_message(conv_id, "assistant", answer_text)
            yield _sse({"type": "done", "conversation_id": conv_id})
        except Exception as exc:  # LLM 调用失败时把错误发给前端，而不是断开
            yield _sse({"type": "error", "detail": str(exc)})

    return StreamingResponse(event_gen(), media_type="text/event-stream")


def _sse(data: dict) -> str:
    return "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"
