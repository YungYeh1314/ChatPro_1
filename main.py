"""ChatPro_1：RAG 问答系统（快速上手提示）。

这个文件不是系统的核心代码，只是一个"说明书"。
运行 `python main.py` 会打印出整个项目需要执行的 3 条命令。
"""


def main():
    """打印上手步骤。

    Python 规定：函数内部的代码只有被调用时才会执行。
    如果本文件被直接运行（python main.py），会走到文件末尾的
    if __name__ == "__main__": 分支并调用这里的 main()。
    """
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


# 这是 Python 的"程序入口"惯用写法：
# 只有直接运行本文件（python main.py）时 __name__ 才是 "__main__"；
# 如果本文件被其他文件 import，则不会自动执行 main()。
if __name__ == "__main__":
    main()
