# ChatPro_1：RAG 问答系统

一个从零实现的 RAG（检索增强生成）问答系统：把语料切块、向量化存入数据库，用户提问时检索最相关的片段，交给大语言模型生成带来源的回答。

## 技术栈

- **数据**：中文维基百科子集（Hugging Face），也可导入任意"标题+正文"格式的本地 JSONL
- **存储**：SQLite（文档/切片/对话）+ Chroma（向量）
- **后端**：FastAPI（流式回答）
- **前端**：Streamlit
- **LLM**：OpenAI 兼容接口（可换 DeepSeek、智谱 GLM 等）

## 架构

```
语料(JSONL) ──> 清洗 ──> 切块 ──> 向量化(embedding)
                                   │
                                   ▼
                          Chroma 向量库 + SQLite
                                   ▲
用户提问 ──> 向量化 ──> 相似度检索（top-k）
                                   │
                    检索到的片段拼进提示词
                                   ▼
                         LLM 生成回答 ──> FastAPI ──> Streamlit
```

## 快速开始

### 0. 准备

```bash
copy .env.example .env
```

默认是 `RAG_MODE=mock` 模式：**不需要 API Key 就能跑通全流程**（用伪向量和规则答案），适合先理解系统。想用真实 LLM 时，把 `.env` 里的 `RAG_MODE` 改成 `real` 并填好密钥。

### 1. 导入数据

方式一：从 Hugging Face 下载中文维基子集（需要网络，约几十 MB 起）：

```bash
python -m app.ingest download --keywords 计算机 编程 人工智能 数据库 --limit 2000
```

方式二：导入本地 JSONL（每条包含 `title` 和 `content` 字段），仓库里已带示例数据：

```bash
python -m app.ingest ingest --file examples/sample_wikipedia.jsonl
```

### 2. 启动后端 API

```bash
uvicorn app.chat_api:app --reload
```

打开 http://127.0.0.1:8000/docs 可以看到接口文档（Swagger）。

### 3. 启动问答界面

另开一个终端：

```bash
streamlit run frontend.py
```

浏览器会自动打开问答界面，输入问题即可。也可以在 Swagger 里直接测试 `/chat` 接口。

## 配置真实 LLM（.env）

```ini
RAG_MODE=real
LLM_API_KEY=sk-xxxxx
LLM_BASE_URL=https://api.deepseek.com/v1   # 以 DeepSeek 为例
LLM_MODEL=deepseek-chat
EMBEDDING_MODEL=text-embedding-3-small     # embedding 模型按供应商选择
EMBEDDING_DIM=1536                          # 与 embedding 模型维度一致
```

> embedding 模型维度必须与 `EMBEDDING_DIM` 一致，且换模型后需要重新导入数据。

## 项目结构

```
ChatPro_1/
├── app/
│   ├── config.py     # 配置（.env）
│   ├── database.py   # SQLite：文档、切片、对话、消息
│   ├── ingest.py     # 下载/导入、清洗、切块、向量化、入库
│   ├── rag.py        # 检索 + 拼提示词 + 调 LLM
│   └── chat_api.py   # FastAPI 接口
├── frontend.py       # Streamlit 问答界面
├── data/             # 语料、SQLite、Chroma（已被 git 忽略）
├── .env.example      # 配置模板
└── requirements.txt
```

## 数据库设计

SQLite 里 4 张表（`data/rag.db`）：

| 表 | 作用 |
|---|---|
| documents | 每篇文档：标题、来源、正文 |
| chunks | 文档被切成的块，`doc_id` 外键指向 documents |
| conversations | 一次对话 |
| messages | 对话中的每条消息（role + content） |

向量存在 Chroma（`data/chroma/`），每条向量通过元数据里的 `doc_id` 与 SQLite 关联。

## 常见问题

**mock 模式回答很机械？** 正常，它只是用来验证流程。配置好 API Key 后设为 `RAG_MODE=real`。

**检索不到相关内容？** 检查是否已导入数据（`python -m app.ingest ingest --file examples/sample_wikipedia.jsonl`），并确认提问内容和语料主题一致。

**换 embedding 模型/维度？** 改 `.env` 后需要清空 `data/chroma/` 并重新导入数据。
