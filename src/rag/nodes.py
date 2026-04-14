"""
src/rag/nodes.py
================
Every node in the LangGraph agentic workflow is defined here as a plain
Python function that takes and returns an ``AgentState`` dict.

Workflow diagram
----------------

  ┌─────────────────────────────────────────────────────────┐
  │                        START                            │
  └──────────────────────┬──────────────────────────────────┘
                         │ question
                         ▼
               ┌──────────────────┐
               │  query_analyzer  │  Classify question & decide retrieval
               └────────┬─────────┘
                        │
          ┌─────────────▼─────────────┐
          │ needs retrieval?           │
          │  YES ──────────────────►  │
          │                     retriever
          │  NO ────────────────────► │
          └─────────────────┬─────────┘
                            │ (NO path skips to generator)
                            ▼
                  ┌──────────────────┐
                  │   retriever      │  Fetch top-k docs from ChromaDB
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ relevance_grader │  Filter docs by relevance
                  └────────┬─────────┘
                           │
              ┌────────────▼───────────┐
              │  relevant docs found?   │
              │  YES ──────────────────►│
              │                   generator
              │  NO (retry budget left?)│
              │  YES → query_rewriter ──┘
              │  NO  → generator (no ctx)│
              └─────────────────────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │    generator     │  Synthesise final answer
                  └────────┬─────────┘
                           │
                         END
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever
from loguru import logger

from src.rag.prompts import (
    GENERATOR_NO_CONTEXT_PROMPT,
    GENERATOR_PROMPT,
    QUERY_ANALYZER_PROMPT,
    QUERY_REWRITER_PROMPT,
    RELEVANCE_GRADER_PROMPT,
)
from src.rag.state import AgentState


# ---------------------------------------------------------------------------
# Node 1: Query Analyser
# ---------------------------------------------------------------------------

def query_analyzer(state: AgentState, llm: BaseChatModel) -> AgentState:
    """
    Uses the LLM to:
    - Classify the question type (installation, api, tutorial, …)
    - Decide whether retrieval is needed
    - Produce an optimised search query

    Updates state keys: query_type, needs_retrieval, search_query
    """
    question = state["question"]
    prompt = QUERY_ANALYZER_PROMPT.format(question=question)

    response = llm.invoke(prompt)
    text = response.content.strip()
    logger.debug(f"[query_analyzer] raw response:\n{text}")

    # Parse structured output
    lines = {
        line.split(":")[0].strip(): line.split(":", 1)[1].strip()
        for line in text.splitlines()
        if ":" in line
    }

    query_type      = lines.get("CATEGORY", "general").lower()
    needs_retrieval = lines.get("NEEDS_RETRIEVAL", "YES").upper() == "YES"
    search_query    = lines.get("SEARCH_QUERY", question)

    logger.info(
        f"[query_analyzer] type={query_type} | "
        f"retrieve={needs_retrieval} | query='{search_query}'"
    )

    return {
        **state,
        "query_type":      query_type,
        "needs_retrieval": needs_retrieval,
        "search_query":    search_query,
    }


# ---------------------------------------------------------------------------
# Node 2: Retriever
# ---------------------------------------------------------------------------

def retriever(state: AgentState, retriever_obj: VectorStoreRetriever) -> AgentState:
    """
    Retrieves the top-k most similar document chunks for the current
    ``search_query`` from ChromaDB.

    Updates state keys: documents
    """
    query = state["search_query"]
    logger.info(f"[retriever] Searching for: '{query}'")

    docs: List[Document] = retriever_obj.invoke(query)
    logger.info(f"[retriever] Retrieved {len(docs)} chunks.")

    return {**state, "documents": docs}


# ---------------------------------------------------------------------------
# Node 3: Relevance Grader
# ---------------------------------------------------------------------------

def relevance_grader(state: AgentState, llm: BaseChatModel) -> AgentState:
    """
    Asks the LLM to score each retrieved chunk as relevant (YES) or not (NO).
    Only keeps chunks graded YES.

    Updates state keys: relevant_docs, is_sufficient
    """
    question  = state["question"]
    documents = state["documents"]

    if not documents:
        logger.warning("[relevance_grader] No documents to grade.")
        return {**state, "relevant_docs": [], "is_sufficient": False}

    relevant: List[Document] = []
    for doc in documents:
        prompt = RELEVANCE_GRADER_PROMPT.format(
            question=question,
            document=doc.page_content[:2000],   # truncate to avoid token overflow
        )
        resp = llm.invoke(prompt)
        grade = resp.content.strip().upper()
        if grade.startswith("YES"):
            relevant.append(doc)

    is_sufficient = len(relevant) > 0
    logger.info(
        f"[relevance_grader] {len(relevant)}/{len(documents)} chunks are relevant."
    )

    return {**state, "relevant_docs": relevant, "is_sufficient": is_sufficient}


# ---------------------------------------------------------------------------
# Node 4: Query Rewriter
# ---------------------------------------------------------------------------

def query_rewriter(state: AgentState, llm: BaseChatModel) -> AgentState:
    """
    If retrieval failed (no relevant docs), rewrite the search query and
    increment the retry counter.

    Updates state keys: search_query, retry_count
    """
    original_query = state["search_query"]
    question       = state["question"]
    retry_count    = state.get("retry_count", 0) + 1

    prompt = QUERY_REWRITER_PROMPT.format(
        query=original_query,
        question=question,
    )
    resp = llm.invoke(prompt)
    new_query = resp.content.strip()

    logger.info(
        f"[query_rewriter] retry #{retry_count}: "
        f"'{original_query}' → '{new_query}'"
    )

    return {
        **state,
        "search_query": new_query,
        "retry_count":  retry_count,
    }


# ---------------------------------------------------------------------------
# Node 5: Generator
# ---------------------------------------------------------------------------

def generator(state: AgentState, llm: BaseChatModel) -> AgentState:
    """
    Synthesises the final answer from the relevant docs (or without context
    if retrieval found nothing after all retries).

    Updates state keys: answer
    """
    question      = state["question"]
    relevant_docs = state.get("relevant_docs", [])

    if relevant_docs:
        # Build context string from relevant chunks
        context_parts = []
        for i, doc in enumerate(relevant_docs, start=1):
            meta   = doc.metadata
            source = meta.get("relative_path", meta.get("source", "unknown"))
            context_parts.append(
                f"[{i}] Source: {source}\n{doc.page_content}"
            )
        context = "\n\n---\n\n".join(context_parts)

        prompt = GENERATOR_PROMPT.format(
            context=context,
            question=question,
        )
    else:
        # No useful context was found — answer from general knowledge
        logger.warning("[generator] Falling back to no-context answer.")
        prompt = GENERATOR_NO_CONTEXT_PROMPT.format(question=question)

    resp   = llm.invoke(prompt)
    answer = resp.content.strip()

    logger.info(f"[generator] Answer generated ({len(answer)} chars).")
    return {**state, "answer": answer}
