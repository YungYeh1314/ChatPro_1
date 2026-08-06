"""ChatPro_1：RAG 问答系统（快速上手提示）。"""


def main():
    print("ChatPro_1 RAG 问答系统")
    print()
    print("1. 导入数据（二选一）")
    print("   python -m app.ingest download   # 从 Hugging Face 下载中文维基子集")
    print("   python -m app.ingest ingest --file 你的文件.jsonl  # 或导入本地语料")
    print("2. 启动后端 API")
    print("   uvicorn app.chat_api:app --reload")
    print("3. 启动问答界面")
    print("   streamlit run frontend.py")
    print()
    print("详细说明见 README.md")


if __name__ == "__main__":
    main()
