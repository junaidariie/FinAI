import os
import logging
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INDEX_NAME = "finance-rag"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RagPipeline:

    def __init__(self, index_name=INDEX_NAME, embedding_model="BAAI/bge-base-en-v1.5"):
        api_key = os.getenv("PINECONE_API_KEY")

        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables.")

        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.bm25_retriever = None
        self.cached_docs = []

        self._ensure_index()

        logger.info("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        logger.info("Embedding model loaded successfully.")

        logger.info("Loading cross-encoder model...")
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        logger.info("Cross-encoder model loaded successfully.")

    def _ensure_index(self):
        existing_indexes = self.pc.list_indexes().names()

        if self.index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {self.index_name}")

            self.pc.create_index(
                name=self.index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

            logger.info("Pinecone index created successfully.")

        else:
            logger.info(f"Pinecone index '{self.index_name}' already exists.")

    def vector_store(self):
        return PineconeVectorStore(
            index=self.pc.Index(self.index_name),
            embedding=self.embeddings
        )

    def load_docs(self, pdf_path: str):
        try:
            logger.info(f"Loading PDF: {pdf_path}")

            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            logger.info(f"Loaded {len(documents)} pages.")

            return documents

        except Exception as e:
            logger.exception("Error loading PDF.")
            raise e

    def split_docs(self, docs, chunk_size=1000, chunk_overlap=250):
        try:
            logger.info("Splitting documents into chunks...")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            split_documents = splitter.split_documents(docs)

            self.cached_docs = split_documents   

            logger.info(f"Generated {len(split_documents)} chunks.")

            return split_documents

        except Exception:
            logger.exception("Error splitting documents.")
            return None

    def add_docs(self, split_docs):
        try:
            logger.info("Uploading chunks to Pinecone...")

            vectorstore = self.vector_store()
            vectorstore.add_documents(split_docs)

            logger.info("Documents uploaded to Pinecone successfully.")

        except Exception:
            logger.exception("Error uploading documents to Pinecone.")

    def delete_all_docs(self):
        try:
            logger.info("Deleting ALL documents from Pinecone index...")

            index = self.pc.Index(self.index_name)

            index.delete(delete_all=True)

            logger.info("All documents deleted successfully.")

            self.bm25_retriever = None
            self.cached_docs = []   

        except Exception:
            logger.exception("Error deleting all documents.")

    def create_bm25(self, split_docs=None, k=4):   
        try:
            logger.info("Creating BM25 retriever...")

            docs = split_docs if split_docs is not None else self.cached_docs

            self.bm25_retriever = BM25Retriever.from_documents(docs)
            self.bm25_retriever.k = k

            logger.info("BM25 retriever ready.")

        except Exception:
            logger.exception("Error creating BM25 retriever.")

    def dense_retriever(self, k=4):
        vectorstore = self.vector_store()

        return vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

    def rerank(self, query: str, docs: list, top_k: int = 6) -> tuple[list, float]:
        """
        Re-rank docs using the cross-encoder and return top_k docs
        along with the highest confidence score (0-1 normalized).
        """
        if not docs:
            return [], 0.0

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.cross_encoder.predict(pairs)

        scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

        top_docs = [doc for _, doc in scored_docs[:top_k]]

        # Normalize top score to 0-1 using sigmoid
        import math
        raw_top_score = float(scored_docs[0][0])
        confidence = round(1 / (1 + math.exp(-raw_top_score)), 4)

        logger.info(f"Cross-encoder reranked {len(docs)} docs → top {len(top_docs)}, confidence={confidence}")

        return top_docs, confidence

    def hybrid_retrieve(self, query, dense_k=6, top_k=6):
        try:
            dense_docs = []

            try:
                dense_docs = self.dense_retriever(k=dense_k).invoke(query)
            except Exception:
                logger.warning("Dense retrieval unavailable.")

            bm25_docs = []

            if self.bm25_retriever is None:
                if self.cached_docs:
                    logger.info("Rebuilding BM25 retriever.")
                    self.create_bm25()
                else:
                    logger.warning("No uploaded docs found. Using direct LLM fallback.")
                    return [], 0.0

            if self.bm25_retriever:
                bm25_docs = self.bm25_retriever.invoke(query)

            combined = bm25_docs + dense_docs

            seen = set()
            unique = []

            for doc in combined:
                text = doc.page_content.strip()
                if text not in seen:
                    seen.add(text)
                    unique.append(doc)

            return self.rerank(query, unique, top_k=top_k)

        except Exception:
            logger.exception("Error during hybrid retrieval.")
            return [], 0.0
