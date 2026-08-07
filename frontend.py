"""Streamlit 问答界面。启动：streamlit run frontend.py

前端是一个独立的浏览器应用，和后端（uvicorn）各自运行、通过 HTTP 通信：
用户输入问题 → requests 请求后端 /chat/stream → 逐字渲染回答 → 展示参考来源。
"""

import json
import os
import requests
import streamlit as st

# 后端地址：优先读环境变量 API_BASE，没有就用本地默认值。
# 后端部署在其他机器/端口时，改这里即可，不用动其他代码。
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

# 页面基本设置：标题和图标
st.set_page_config(page_title="ChatPro RAG 问答", page_icon="🤖")
st.title("🤖 ChatPro RAG 问答系统")

# ---- 初始化会话状态 ----
# st.session_state 是 Streamlit 的"内存变量"：脚本每次交互都会从头重跑，
# 普通局部变量会丢失，只有放在 session_state 里的数据能跨交互保留。
# messages：聊天记录 [{"role": ..., "content": ...}, ...]
if "messages" not in st.session_state:
    st.session_state.messages = []
# conversation_id：当前会话 id，用于多轮对话时后端能查到历史消息
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

# ---- 渲染历史消息 ----
# 每次重跑脚本时，把已保存的历史消息重新画一遍，聊天记录才不会消失。
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):  # 根据 role 决定气泡显示在左侧还是右侧
        st.markdown(msg["content"])

# ---- 处理新输入 ----
# 海象运算符 :=：把 st.chat_input 的结果赋值给 question，同时判断是否非空。
# 只有用户输入了问题并回车，下面整段代码才会执行。
if question := st.chat_input("输入你的问题"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 助手回答区：先放一个空占位，随着流式数据不断刷新内容，形成打字机效果
    with st.chat_message("assistant"):
        placeholder = st.empty()  # 可反复更新内容的空占位元素
        full = ""                 # 累积的完整回答（流式结束后用于存档）
        sources = []              # 检索到的参考来源
        try:
            # 向后端发起流式 POST 请求
            # stream=True：不等待完整响应，边收边处理
            with requests.post(
                f"{API_BASE}/chat/stream",
                json={
                    "question": question,
                    "conversation_id": st.session_state.conversation_id,
                },
                stream=True,
                timeout=120,  # 最长等待 120 秒，防止请求挂死
            ) as resp:
                # 逐行读取 SSE 事件（每行格式："data: {...}"）
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue  # 跳过空行和非数据行（如心跳、空事件）
                    # 去掉 "data: " 前缀，剩下的就是 JSON
                    event = json.loads(line[len("data: ") :])
                    if event["type"] == "sources":
                        sources = event["sources"]  # 后端先发来源
                    elif event["type"] == "token":
                        full += event["content"]         # 追加一个字/词
                        placeholder.markdown(full + "▌")  # 光标符号制造"正在打字"感
                    elif event["type"] == "done":
                        # 后端已把完整回答存库，回传 conversation_id 供下次使用
                        st.session_state.conversation_id = event["conversation_id"]
                    elif event["type"] == "error":
                        full = f"出错了：{event['detail']}"
        except Exception as exc:
            # 网络不通、后端没启动等情况会走到这里
            full = f"请求失败（后端启动了吗？）：{exc}"
        placeholder.markdown(full)  # 去掉光标符号，显示最终回答

        # ---- 展示参考来源 ----
        if sources:
            # st.expander：可折叠区域，点击展开查看来源
            with st.expander(f"📚 参考来源（{len(sources)} 条）"):
                for i, s in enumerate(sources, 1):
                    st.markdown(f"**{i}. {s['title']}**（相似度 {s['score']:.2f}）")
                    st.text(s["content"][:300])  # 截取前 300 字做预览

    # 把完整回答存进会话状态，下次重跑脚本时能显示出来
    st.session_state.messages.append({"role": "assistant", "content": full})
