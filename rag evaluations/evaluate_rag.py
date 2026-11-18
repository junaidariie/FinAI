import argparse
import csv
import importlib
import json
import logging
import math
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "finance_rag_20_question_answer.pdf"
OUTPUT_DIR = Path(__file__).resolve().parent
METRIC_COLUMNS = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "context_utilization",
)
JUDGE_MODEL = "mistralai/mistral-small-2603"
JUDGE_BASE_URL = "https://api.xkiro.com/v1"
QA_PATTERN = re.compile(
    r"(?m)^\s*(\d+)\.\s*Question:\s*(.*?)\s*Answer:\s*(.*?)"
    r"(?=^\s*\d+\.\s*Question:|\Z)",
    re.DOTALL,
)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_qa_pairs(pdf_path: Path) -> List[Dict[str, str]]:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Evaluation PDF not found: {pdf_path}")

    text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages)
    pairs = []
    for match in QA_PATTERN.finditer(text):
        question = re.sub(r"\s+", " ", match.group(2)).strip()
        answer = re.sub(r"\s+", " ", match.group(3)).strip()
        if question and answer:
            pairs.append(
                {
                    "question_number": match.group(1),
                    "question": question,
                    "reference_answer": answer,
                }
            )

    if not pairs:
        raise ValueError(f"No Question/Answer pairs could be parsed from {pdf_path}.")
    return pairs


def _wait_for_upload(pipeline: Any, expected_count: int, timeout_seconds: int = 90) -> None:
    index = pipeline.pc.Index(pipeline.index_name)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        stats = index.describe_index_stats()
        namespace_stats = stats.namespaces.get(pipeline.namespace)
        vector_count = getattr(namespace_stats, "vector_count", 0)
        if vector_count >= expected_count:
            return
        time.sleep(2)
    raise RuntimeError(
        f"Expected {expected_count} vectors in the temporary evaluation namespace, "
        "but the upload did not complete."
    )


def _wait_for_namespace_empty(pipeline: Any, timeout_seconds: int = 90) -> None:
    index = pipeline.pc.Index(pipeline.index_name)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        stats = index.describe_index_stats()
        namespace_stats = stats.namespaces.get(pipeline.namespace)
        if getattr(namespace_stats, "vector_count", 0) == 0:
            return
        time.sleep(2)
    raise RuntimeError(f"Evaluation namespace {pipeline.namespace!r} was not emptied.")


def evaluate(pdf_path: Path, limit: int, output_dir: Path) -> Path:
    if limit < 1:
        raise ValueError("--limit must be at least 1.")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    load_dotenv(dotenv_path=ROOT / ".env")

    from pinecone import Pinecone

    from openai import AsyncOpenAI
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        ContextUtilization,
        Faithfulness,
    )

    from scripts.rag import INDEX_NAME, RagPipeline

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY is required for hybrid retrieval.")
    judge_api_key = os.getenv("XKIRO_API_KEY")
    if not judge_api_key:
        raise ValueError("XKIRO_API_KEY is required for Ragas evaluation.")

    qa_pairs = load_qa_pairs(pdf_path)[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    namespace = f"finai-eval-{uuid.uuid4().hex[:12]}"
    pinecone_client = Pinecone(api_key=api_key)
    if INDEX_NAME not in pinecone_client.list_indexes().names():
        raise RuntimeError(
            f"Pinecone index {INDEX_NAME!r} must exist before evaluation can use a namespace."
        )
    pipeline = None
    namespace_isolated = False

    try:
        pipeline = RagPipeline(index_name=INDEX_NAME, namespace=namespace)
        namespace_stats = pipeline.pc.Index(INDEX_NAME).describe_index_stats().namespaces.get(
            namespace
        )
        if getattr(namespace_stats, "vector_count", 0):
            raise RuntimeError("The generated evaluation namespace is unexpectedly in use.")
        namespace_isolated = True

        pages = pipeline.load_docs(str(pdf_path))
        split_documents = pipeline.split_docs(pages)
        if not split_documents:
            raise RuntimeError("The evaluation PDF did not produce any document chunks.")

        pipeline.add_docs(split_documents)
        pipeline.create_bm25(split_documents)
        if pipeline.bm25_retriever is None:
            raise RuntimeError("BM25 retriever could not be initialized from the PDF.")
        _wait_for_upload(pipeline, len(split_documents))

        chat_pipeline = importlib.import_module("scripts.main")
        chat_pipeline.set_rag_instance(pipeline)
        if chat_pipeline.llm is None:
            raise RuntimeError("The fine-tuned model did not load; see scripts/load_llm.py logs.")

        rows = []
        ragas_samples = []
        for pair in qa_pairs:
            chunks = list(
                chat_pipeline.stream_chat_response(
                    user_message=pair["question"],
                    thread_id=f"evaluation-{pair['question_number']}-{uuid.uuid4().hex}",
                )
            )
            generated_answer = "".join(str(chunk.get("token", "")) for chunk in chunks).strip()
            if not generated_answer:
                raise RuntimeError(
                    f"The model returned an empty answer for question {pair['question_number']}."
                )

            metadata = chunks[0].get("metadata", {}) if chunks else {}
            context = metadata.get("context", "")
            retrieved_contexts = metadata.get("retrieved_contexts", [])
            if not context or not retrieved_contexts:
                raise RuntimeError(
                    f"No context was retrieved for question {pair['question_number']}."
                )

            rows.append(
                {
                    **pair,
                    "generated_answer": generated_answer,
                    "retrieved_context": context,
                    "retrieved_contexts": json.dumps(retrieved_contexts, ensure_ascii=False),
                    "retrieved_chunks": metadata.get("retrieved_chunks", 0),
                    "retrieval_confidence": metadata.get("confidence", 0.0),
                }
            )
            ragas_samples.append(
                {
                    "user_input": pair["question"],
                    "response": generated_answer,
                    "retrieved_contexts": retrieved_contexts,
                    "reference": pair["reference_answer"],
                }
            )
            logger.info("Evaluated question %s.", pair["question_number"])

        evaluator_client = AsyncOpenAI(
            api_key=judge_api_key,
            base_url=JUDGE_BASE_URL,
        )
        evaluator_llm = llm_factory(JUDGE_MODEL, client=evaluator_client)
        evaluator_embeddings = HuggingFaceEmbeddings(
            model="BAAI/bge-small-en-v1.5"
        )
        context_precision = ContextPrecision(llm=evaluator_llm)
        context_recall = ContextRecall(llm=evaluator_llm)
        faithfulness = Faithfulness(llm=evaluator_llm)
        answer_relevancy = AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
        context_utilization = ContextUtilization(llm=evaluator_llm)
        for row, sample in zip(rows, ragas_samples):
            scores = {
                "context_precision": context_precision.score(
                    user_input=sample["user_input"],
                    reference=sample["reference"],
                    retrieved_contexts=sample["retrieved_contexts"],
                ).value,
                "context_recall": context_recall.score(
                    user_input=sample["user_input"],
                    reference=sample["reference"],
                    retrieved_contexts=sample["retrieved_contexts"],
                ).value,
                "faithfulness": faithfulness.score(
                    user_input=sample["user_input"],
                    response=sample["response"],
                    retrieved_contexts=sample["retrieved_contexts"],
                ).value,
                "answer_relevancy": answer_relevancy.score(
                    user_input=sample["user_input"],
                    response=sample["response"],
                ).value,
                "context_utilization": context_utilization.score(
                    user_input=sample["user_input"],
                    response=sample["response"],
                    retrieved_contexts=sample["retrieved_contexts"],
                ).value,
            }
            for name in METRIC_COLUMNS:
                value = float(scores[name])
                if not math.isfinite(value):
                    raise RuntimeError(f"Ragas produced an invalid {name} score.")
                row[name] = round(value, 4)

        results_path = output_dir / "results.csv"
        columns = (
            "question_number",
            "question",
            "reference_answer",
            "generated_answer",
            "retrieved_context",
            "retrieved_contexts",
            "retrieved_chunks",
            "retrieval_confidence",
            *METRIC_COLUMNS,
        )
        with results_path.open("w", newline="", encoding="utf-8-sig") as results_file:
            writer = csv.DictWriter(results_file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

        from plot_results import plot_metrics

        plot_metrics(results_path, output_dir / "metrics.png")
        logger.info("Saved evaluation results to %s.", results_path)
        return results_path
    finally:
        if pipeline is not None and namespace_isolated:
            pipeline.delete_all_docs()
            _wait_for_namespace_empty(pipeline)
            logger.info("Removed evaluation vectors from namespace %s.", namespace)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the fine-tuned FinAI RAG pipeline with Ragas metrics."
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    evaluate(args.pdf, args.limit, args.output_dir)


if __name__ == "__main__":
    main()
