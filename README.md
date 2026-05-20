# FinAI — Financial Intelligence Assistant

A fine-tuned **Qwen 2.5 (1.5B)** model for financial question answering, served via a **FastAPI** backend with a **Hybrid RAG** pipeline powered by **Pinecone** and **LangGraph**.

---

## 📓 Training Notebook &nbsp;|&nbsp; 🎥 Demo

> GitHub does not render large Jupyter notebooks. The full fine-tuning notebook is hosted on Hugging Face for proper viewing.

**[View Training Notebook on Hugging Face →](https://huggingface.co/spaces/junaid17/FinAI/blob/main/Qwen2_5_financial_finetuning.ipynb)**

[![FinAI Demo](https://img.youtube.com/vi/H0qx3JrcYv8/0.jpg)](https://youtu.be/H0qx3JrcYv8)

---

## 🧠 Model Fine-Tuning

### Base Model
- **Qwen 2.5 — 1.5B parameters**

### Datasets Used
| Dataset | Description |
|---|---|
| [`LLukas22/fiqa`](https://huggingface.co/datasets/LLukas22/fiqa) | Financial QA pairs from community forums |
| [`gbharti/finance-alpaca`](https://huggingface.co/datasets/gbharti/finance-alpaca) | Instruction-following financial dataset |

### Fine-Tuning Method — QLoRA with Unsloth
Used **LoRA (Low-Rank Adaptation)** via `unsloth`'s `FastLanguageModel` for memory-efficient fine-tuning:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
```

---

## 📊 Evaluation Results

Significant improvements across all metrics after fine-tuning:

| Metric | Base Model | Fine-Tuned | Result |
|---|---|---|---|
| PPL ↓ | 28.0700 | **5.8100** | ✅ improved |
| BLEU ↑ | 2.3573 | **3.8204** | ✅ improved |
| ROUGE-1 ↑ | 0.2596 | **0.3069** | ✅ improved |
| ROUGE-L ↑ | 0.1439 | **0.1805** | ✅ improved |

### Metric Plots

![Metrics Comparison](assets/metrices.png)

### Improvements Overview

![Improvements](assets/improvements.png)

---

## 🏗️ System Architecture

### LangGraph Workflow

The backend uses **LangGraph** to orchestrate a stateful RAG pipeline with conditional routing:

![LangGraph Workflow](assets/workflow.png)

**Flow:**
1. `retrieve` — Hybrid retrieval (Dense + BM25) from Pinecone
2. `relevance` — Checks if retrieved docs are meaningful (>50 chars)
3. **Conditional Route:**
   - If relevant docs found → `build_context` → `rag_prompt` → LLM
   - If no relevant docs → `direct_prompt` → LLM

---

## 🔌 FastAPI Backend

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/upload` | Upload a PDF document for RAG |
| `DELETE` | `/reset` | Delete all documents from the vector store |
| `POST` | `/chat/stream` | Streaming chat with SSE (Server-Sent Events) |

### `/chat/stream` — Streaming Response Format

The chat endpoint streams responses using **Server-Sent Events (SSE)**. Three event types are emitted:

```
data: {"type": "metadata", "used_rag": true, "sources": [...], "thread_id": "..."}

data: {"type": "token", "content": "..."}

data: {"type": "done"}
```

### `/upload` — Document Ingestion Flow

```
PDF Upload → Delete old docs → Load → Split (1000 chars, 250 overlap)
          → Embed & store in Pinecone → Build BM25 index
```

---

## 🔍 RAG Pipeline

### Embedding Model — `BAAI/bge-base-en-v1.5`

Chosen for its excellent balance of **speed and retrieval quality**:
- Ranked highly on the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for retrieval tasks
- 768-dimensional embeddings — compact yet highly expressive
- Significantly faster inference than larger models (e.g., `bge-large`) with minimal quality trade-off
- Optimized for semantic similarity, making it ideal for financial document retrieval

### Vector Database — Pinecone

- **Serverless** index on AWS `us-east-1`
- Cosine similarity metric
- 768-dimensional vectors matching the embedding model

### Hybrid Retriever — Dense + BM25

Combines two complementary retrieval strategies:

| Retriever | Type | Strength |
|---|---|---|
| **Dense** (Pinecone) | Semantic / vector search | Understands meaning and context |
| **BM25** | Keyword / lexical search | Exact term matching, financial jargon |

Results are merged and deduplicated, giving the best of both worlds — semantic understanding and precise keyword matching critical for financial terminology.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Pinecone account
- `llama-cpp-python` installed (with appropriate backend for your hardware)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

```env
PINECONE_API_KEY=<your_pinecone_api_key>
```

### Run the Server

```bash
uvicorn app:app --reload
```

API will be live at `http://localhost:8000`

---

## 🗂️ Project Structure

```
FINAL-CODER/
├── assets/
│   ├── workflow.png        # LangGraph workflow diagram
│   ├── metrices.png        # Metric comparison plots
│   └── improvements.png    # Before/after improvement chart
├── scripts/
│   ├── load_llm.py         # Model loader (ChatLlamaCpp singleton)
│   ├── main.py             # LangGraph graph + streaming logic
│   └── rag.py              # RAG pipeline (Pinecone + BM25)
├── Notebook/
│   └── Qwen2_5_financial_finetuning.ipynb
├── app.py                  # FastAPI application
├── requirements.txt
└── .env
```

---

## 🤗 Model on Hugging Face

The fine-tuned GGUF model is hosted on Hugging Face:

**[junaid17/qwen2.5-finance-assistant-gguf](https://huggingface.co/junaid17/qwen2.5-finance-assistant-gguf)**

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Qwen 2.5 1.5B (QLoRA fine-tuned, GGUF) |
| Inference | `llama-cpp-python` via `ChatLlamaCpp` |
| Fine-tuning | Unsloth + QLoRA |
| Orchestration | LangGraph |
| Vector DB | Pinecone (Serverless) |
| Embeddings | `BAAI/bge-base-en-v1.5` |
| Sparse Retrieval | BM25 (`rank-bm25`) |
| Backend | FastAPI + Uvicorn |
| Memory | LangGraph `MemorySaver` (per thread_id) |
