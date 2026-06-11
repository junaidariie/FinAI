from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from scripts.rag import RagPipeline
from scripts.load_llm import get_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver


# Initialized Rag and llm
RAG = None
llm = get_model()

def set_rag_instance(rag_instance):
    global RAG
    RAG = rag_instance

# Initializing ChatState
class ChatState(TypedDict):
    query: str
    retrieved_docs: list
    context: str
    use_rag: bool
    final_prompt: str
    sources: list
    confidence: float


# Making Nodes

def retrieve_node(state):
    query = state["query"]

    docs, confidence = RAG.hybrid_retrieve(query=query, dense_k=6, top_k=6)

    return {"retrieved_docs": docs, "confidence": confidence}


def relevance_node(state):
    docs = state["retrieved_docs"]

    use_rag = False

    if docs:
        meaningful_docs = [
            d for d in docs
            if len(d.page_content.strip()) > 50
        ]

        use_rag = len(meaningful_docs) > 0

    return {"use_rag": use_rag}


def build_context_node(state):
    docs = state["retrieved_docs"]

    context = ""
    sources = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")

        sources.append({
            "document": source,
            "page": page
        })

        context += f"""
        SOURCE: {source}
        PAGE: {page}

        CONTENT:
        {doc.page_content}
        """

    return {
        "context": context,
        "sources": sources
    }


def rag_prompt_node(state):
    query = state["query"]
    context = state["context"]

    prompt = f"""
        You are a financial intelligence assistant.

        Use ONLY the provided context.

        If context is insufficient, say so.

        Context:
        {context}

        Question:
        {query}
        """

    return {
        "final_prompt": prompt
    }

def direct_prompt_node(state):
    query = state["query"]

    prompt = f"""
        You are a financial intelligence assistant.
        your job is to answer the user's question to the best of your ability.

        Question:
        {query}
        """

    return {
        "final_prompt": prompt,
        "sources": [],
        "confidence": 0.0
    }

def route_decision(state):
    if state["use_rag"]:
        return "build_context"

    return "direct_prompt"



memory = MemorySaver()
workflow = StateGraph(ChatState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("relevance", relevance_node)
workflow.add_node("build_context", build_context_node)
workflow.add_node("rag_prompt", rag_prompt_node)
workflow.add_node("direct_prompt", direct_prompt_node)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "relevance")

workflow.add_conditional_edges(
    "relevance",
    route_decision,
    {
        "build_context": "build_context",
        "direct_prompt": "direct_prompt"
    }
)

workflow.add_edge("build_context", "rag_prompt")
workflow.add_edge("rag_prompt", END)
workflow.add_edge("direct_prompt", END)

graph_app = workflow.compile(checkpointer=memory)

def stream_chat_response(user_message: str, thread_id: str):
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    state = graph_app.invoke(
        {"query": user_message},
        config=config
    )

    metadata = {
        "used_rag": state["use_rag"],
        "sources": state["sources"],
        "confidence": state.get("confidence", 0.0),
        "thread_id": thread_id
    }

    for chunk in llm.stream(
        [HumanMessage(content=state["final_prompt"])]
    ):
        if chunk.content:
            yield {
                "token": chunk.content,
                "metadata": metadata
            }
