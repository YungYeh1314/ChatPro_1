"""Streamlit 问答界面。启动：streamlit run frontend.py"""

import json
import os

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="ChatPro RAG 问答", page_icon="🤖")
st.title("🤖 ChatPro RAG 问答系统")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("输入你的问题"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""
        sources = []
        try:
            with requests.post(
                f"{API_BASE}/chat/stream",
                json={
                    "question": question,
                    "conversation_id": st.session_state.conversation_id,
                },
                stream=True,
                timeout=120,
            ) as resp:
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    event = json.loads(line[len("data: ") :])
                    if event["type"] == "sources":
                        sources = event["sources"]
                    elif event["type"] == "token":
                        full += event["content"]
                        placeholder.markdown(full + "▌")
                    elif event["type"] == "done":
                        st.session_state.conversation_id = event["conversation_id"]
                    elif event["type"] == "error":
                        full = f"出错了：{event['detail']}"
        except Exception as exc:
            full = f"请求失败（后端启动了吗？）：{exc}"
        placeholder.markdown(full)

        if sources:
            with st.expander(f"📚 参考来源（{len(sources)} 条）"):
                for i, s in enumerate(sources, 1):
                    st.markdown(f"**{i}. {s['title']}**（相似度 {s['score']:.2f}）")
                    st.text(s["content"][:300])

    st.session_state.messages.append({"role": "assistant", "content": full})
