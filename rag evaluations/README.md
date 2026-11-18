# FinAI RAG evaluation

Run these commands from the repository root. Start with one question:

```powershell
python "rag evaluations\evaluate_rag.py" --limit 1
```

Evaluate all 20 question-answer pairs:

```powershell
python "rag evaluations\evaluate_rag.py" --limit 20
```

The evaluator loads `finance_rag_20_question_answer.pdf`, indexes its chunks
in a temporary namespace in the existing `finance-rag` Pinecone index, and
retrieves context with the project's `RagPipeline` hybrid retriever (dense
search, BM25, and reranking). The existing LangGraph chat pipeline generates
answers with the fine-tuned Qwen model from `scripts/load_llm.py`.

Ragas scores Context Precision, Context Recall, Faithfulness, Answer Relevancy,
and Context Utilization. The judge is `mistralai/mistral-small-2603` via the
OpenAI-compatible XKIRO endpoint; set `XKIRO_API_KEY` in `.env`. Answer
Relevancy uses Ragas' `BAAI/bge-small-en-v1.5` embeddings.

The per-question scores are written to `results.csv`; `metrics.png` plots their
average. To regenerate the plot from an existing CSV:

```powershell
python "rag evaluations\plot_results.py"
```

Only the temporary evaluation namespace is deleted after a run; the existing
`finance-rag` namespace is left untouched. Since this PDF contains the reference
answers and is also the retrieval corpus, scores reflect self-retrieval rather
than a held-out document benchmark.
