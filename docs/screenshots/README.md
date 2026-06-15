# 展示截图清单

本目录用于保存真实运行截图。不要放伪造截图；截图应来自本机实际运行的 FastAPI、Streamlit、Qdrant 或推理网关页面。

## 启动地址

| 服务 | 地址 |
|---|---|
| FastAPI Swagger | `http://127.0.0.1:8000/docs` |
| Streamlit 前端 | `http://localhost:8501` |
| Qdrant Dashboard | `http://127.0.0.1:6333/dashboard` |
| 推理网关 Swagger | `http://127.0.0.1:8010/docs` |

## 建议截图

| 文件名 | 页面 | 要展示的内容 |
|---|---|---|
| `01_swagger_api.png` | FastAPI Swagger | `/chat`、`/documents`、`/agent` 接口 |
| `02_rag_chat.png` | Streamlit RAG 问答 | answer_mode、rewritten_query、answer |
| `03_sources_detail.png` | sources 展开 | file_name、page、score、rerank_score、final_score、retrieval_source |
| `04_rag_eval.png` | RAG 评测页 | source_hit、answer_mode、Bad Case 表格 |
| `05_agent_report.png` | 论文调研 Agent 页 | plan、tool_calls、evidence_items、final_report |
| `06_gateway_logs.png` | 推理网关日志/metrics | RAG 调用 Chat/Embedding 的 request_id 或 metrics |

## 截图前检查

- Qdrant、Ollama、推理网关、FastAPI、Streamlit 均已启动。
- 知识库至少入库一篇公开论文。
- `/chat` 返回 `rewritten_query` 和 sources。
- 评测页能跑出至少一条结果。
- 不展示手机号、邮箱、私有论文或未公开材料。
