"""ChatPro_1 RAG 问答系统。

__init__.py 的作用：
1. 告诉 Python 这个目录是一个"包"（package），可以被 import；
2. 包被导入时会执行这里的内容（通常只放说明文字或公共导出）。

正因为有这个文件，app/ 下的模块之间才能用 `from . import config`
这种相对导入写法，也才能运行 `python -m app.ingest` 这类命令。
"""
