"""
src/rag/agent.py
================
Assembles the LangGraph StateGraph that wires together all nodes
into the agentic RAG workflow.

The graph implements the following control flow:

  START
    │
    ▼
  query_analyzer          ← classifies question, decides retrieval
    │
    ├─ needs_retrieval=True  ──► retriever ──► relevance_grader
    │                                                   │
    │                        ┌──── is_sufficient=True ──┘
    │                        │
    │                        ├─── is_sufficient=False & retries < max
    │                        │         └─► query_rewriter ──► retriever
    │                        │
    │                        └─── is_sufficient=False & retries >= max
    │                                    └─► generator (no context)
    │
    └─ needs_retrieval=False ──► generator (no context)

  generator ──► END

Usage
-----
    from src.rag.agent import build_agent
    agent = build_agent(llm, retriever, max_retries=2)
    result = agent.invoke({"question": "How do I install ITHACA-FV?"})
    print(result["answer"])
"""

from functools import partial
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph import END, START, StateGraph
from loguru import logger

from src.rag.nodes import (
    generator,
    query_analyzer,
    query_rewriter,
    relevance_grader,
    retriever,
)
from src.rag.state import AgentState


# ---------------------------------------------------------------------------
# Edge routing functions (pure functions — no LLM calls)
# ---------------------------------------------------------------------------

def _route_after_analyzer(
    state: AgentState,
) -> Literal["retriever", "generator"]:
    """
    After query_analyzer:
    - Go to retriever if retrieval is needed
    - Go directly to generator if retrieval is not needed
    """
    if state["needs_retrieval"]:
        return "retriever"
    return "generator"


def _route_after_grader(
    state: AgentState,
    max_retries: int,
) -> Literal["generator", "query_rewriter"]:
    """
    After relevance_grader:
    - If relevant docs were found → generator
    - If no relevant docs and retries remain → query_rewriter
    - If no relevant docs and retries exhausted → generator (no context)
    """
    if state["is_sufficient"]:
        logger.debug("[router] Sufficient context found → generator")
        return "generator"

    if state.get("retry_count", 0) < max_retries:
        logger.debug(
            f"[router] No relevant docs, retry "
            f"{state.get('retry_count', 0)+1}/{max_retries} → query_rewriter"
        )
        return "query_rewriter"

    logger.warning("[router] Max retries reached → generator (no context)")
    return "generator"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_agent(
    llm: BaseChatModel,
    retriever_obj: VectorStoreRetriever,
    max_retries: int = 2,
):
    """
    Build and compile the LangGraph StateGraph.

    Parameters
    ----------
    llm:          any LangChain chat model (Ollama, vLLM, OpenRouter, …)
    retriever_obj: ChromaDB retriever from ``get_retriever()``
    max_retries:  how many query-rewrite attempts before giving up

    Returns
    -------
    CompiledStateGraph — call ``.invoke({"question": "..."})`` to run
    """
    # -----------------------------------------------------------------------
    # Bind each node function to its dependencies via functools.partial
    # This keeps node signatures compatible with LangGraph (state → state)
    # -----------------------------------------------------------------------
    _query_analyzer   = partial(query_analyzer,   llm=llm)
    _retriever        = partial(retriever,         retriever_obj=retriever_obj)
    _relevance_grader = partial(relevance_grader,  llm=llm)
    _query_rewriter   = partial(query_rewriter,    llm=llm)
    _generator        = partial(generator,         llm=llm)

    # Edge router also needs max_retries baked in
    _route_grader = partial(_route_after_grader, max_retries=max_retries)

    # -----------------------------------------------------------------------
    # Define the graph
    # -----------------------------------------------------------------------
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("query_analyzer",   _query_analyzer)
    graph.add_node("retriever",        _retriever)
    graph.add_node("relevance_grader", _relevance_grader)
    graph.add_node("query_rewriter",   _query_rewriter)
    graph.add_node("generator",        _generator)

    # Entry point
    graph.add_edge(START, "query_analyzer")

    # After analyzer: go to retriever or skip to generator
    graph.add_conditional_edges(
        "query_analyzer",
        _route_after_analyzer,
        {"retriever": "retriever", "generator": "generator"},
    )

    # Retriever always goes to grader
    graph.add_edge("retriever", "relevance_grader")

    # After grader: go to generator or rewriter
    graph.add_conditional_edges(
        "relevance_grader",
        _route_grader,
        {"generator": "generator", "query_rewriter": "query_rewriter"},
    )

    # Rewriter loops back to retriever
    graph.add_edge("query_rewriter", "retriever")

    # Generator always ends the workflow
    graph.add_edge("generator", END)

    # -----------------------------------------------------------------------
    # Compile and return
    # -----------------------------------------------------------------------
    compiled = graph.compile()
    logger.success("LangGraph agent compiled successfully.")
    return compiled


# ---------------------------------------------------------------------------
# Factory: load everything and return a ready-to-use agent
# ---------------------------------------------------------------------------

def load_agent(config: dict):
    """
    High-level factory that reads the YAML config dict, initialises all
    components, and returns a compiled agent.

    Parameters
    ----------
    config: the dict loaded from ``config/config.yaml``

    Returns
    -------
    CompiledStateGraph
    """
    from src.embeddings.embedder import LocalEmbedder
    from src.rag.llm_factory import get_llm
    from src.vectorstore.chroma_store import get_retriever, load_vectorstore

    # 1. Embedding model
    emb_cfg  = config["embeddings"]
    embedder = LocalEmbedder(
        model_name=emb_cfg["model"],
        device=emb_cfg.get("device", "cuda"),
        batch_size=emb_cfg.get("batch_size", 64),
    )

    # 2. Vector store + retriever
    vs_cfg      = config["vectorstore"]
    rag_cfg     = config["rag"]
    vectorstore = load_vectorstore(
        embedder=embedder,
        persist_dir=vs_cfg["persist_dir"],
        collection_name=vs_cfg["collection_name"],
    )
    retriever_obj = get_retriever(
        vectorstore=vectorstore,
        top_k=rag_cfg["top_k"],
        score_threshold=rag_cfg.get("score_threshold", 0.35),
    )

    # 3. LLM
    llm = get_llm(config["llm"])

    # 4. Build agent
    return build_agent(
        llm=llm,
        retriever_obj=retriever_obj,
        max_retries=rag_cfg.get("max_retries", 2),
    )
