# RAG 评测基线

`scripts/evaluate_rag.py` 会执行当前知识库问答链路，并生成三类结果：

```powershell
python scripts\evaluate_rag.py --limit 3
```

真实运行前先做依赖和服务预检，避免把“服务未启动”误写成模型效果：

```powershell
python scripts\preflight_rag_eval.py --provider rule
```

规则 Rerank 与 Cross-Encoder 的隔离 A/B 运行：

```powershell
python scripts\run_rag_ab.py --providers rule,cross_encoder
```

每次评测会写入 `rag_eval_manifest.json`，记录模型、Rerank、Qdrant collection、dense/sparse 配置、chunk 参数、运行平台和 Git revision。A/B 结果位于 `reports/rag_ab/`。只有在 Qdrant、Ollama/推理网关和真实 Cross-Encoder 都可用时，才可以把 A/B 数字作为项目效果证据。

在修改切分参数、解析器或样本后，先校验 gold 标注：

```powershell
python scripts\validate_eval_ground_truth.py
```

输出到 `reports/`：

- `rag_eval_results.csv`：逐问题的 query、答案模式、来源、分数、指标和延迟。
- `project_metrics.md`：适合截图和面试讲解的汇总报告。
- `rag_eval_summary.json`：适合后续 CI、Dashboard 或回归比较的机器可读结果。
- `rag_eval_manifest.json`：记录当前评测的可复现配置和环境信息。

## QA 数据格式

`data/eval/qa_pairs.csv` 保留原有字段即可运行：

```csv
question,expected_answer,source_file,difficulty
```

为了从“文件级命中”升级到“证据级命中”，可以增加以下可选字段：

```csv
gold_source_files,gold_chunk_ids,gold_pages,top_k
paper_a.pdf;paper_b.pdf,paper_a_p3_c02;paper_b_p5_c01,3;5,5
```

多个值使用 `;` 或 `|` 分隔。评测精度按以下顺序选择：

1. `gold_chunk_ids`：精确到 chunk。
2. `gold_pages` + `source_file`：精确到文件和页码。
3. `source_file`：兼容现有数据的文件级来源判断。

## 指标含义

- `Retrieval Precision@K`：返回的 Top-K 中有多少是标注的相关证据。
- `Retrieval Recall@K`：标注证据有多少被 Top-K 找到。
- `MRR`：第一个相关证据出现位置的倒数。
- `Citation Precision/Recall@K`：最终 grounded 回答实际展示的 sources 的覆盖质量。
- `citation_marker_invalid`：grounded 回答没有生成 `[S1]` 形式标记，或引用了没有返回的 source ID；它是引用契约告警，不等于答案语义错误。
- `Answer Token F1`：答案与 `expected_answer` 的词面重叠回归指标，不等于语义正确率。
- `P95 latency`：评测请求耗时的高分位指标。

答案语义正确率仍应通过人工复核或后续的语义 judge 补充，不能只看 Token F1。

## 结构化引用契约

grounded Prompt 会把每个检索片段编号为 `[S1]`、`[S2]`……，API 的 `sources` 和 `citations` 使用同一编号。评测脚本会检查回答中的编号是否存在，并把问题写入 `citation_warnings` 和 `bad_case_type`。fallback/no_answer 不强制要求知识库引用，因为它们不应伪造 sources。

## 运行顺序

1. 启动 Qdrant、Ollama 或推理网关，并确认 Embedding/Chat 模型可用。
2. 如果是旧 dense 索引，运行 `python scripts\migrate_sparse_index.py` 补齐 native sparse sidecar。
3. 运行 `python scripts\validate_eval_ground_truth.py`，确认切分变化没有使 gold chunk/page 失效。
4. 运行 `python scripts\evaluate_rag.py --run-label rule`，保存 baseline。
5. 安装 `requirements-reranker.txt` 后运行 `python scripts\run_rag_ab.py`，对比 retrieval、citation、answer overlap 和延迟。
