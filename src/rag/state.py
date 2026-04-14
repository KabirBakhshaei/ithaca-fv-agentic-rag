"""
src/rag/state.py
================
Defines the ``AgentState`` TypedDict — the single shared data structure
that flows through every node in the LangGraph workflow.

Think of it as a "blackboard" that nodes read from and write to.
Each node receives the full state, makes changes, and returns the updated state.
"""

from typing import List

from langchain_core.documents import Document
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Fields
    ------
    question       : original user question (never modified after init)
    query_type     : classification from query_analyzer
                     ("installation" | "api" | "tutorial" | "theory" |
                      "troubleshooting" | "general")
    search_query   : current search query (may be rewritten by query_rewriter)
    needs_retrieval: whether the workflow should fetch docs
    documents      : raw docs returned by the retriever (unfiltered)
    relevant_docs  : docs graded as relevant by the relevance_grader
    is_sufficient  : True if at least one relevant doc was found
    retry_count    : how many query-rewrite retries have occurred
    answer         : final generated answer (set by generator)
    """

    question:        str
    query_type:      str
    search_query:    str
    needs_retrieval: bool
    documents:       List[Document]
    relevant_docs:   List[Document]
    is_sufficient:   bool
    retry_count:     int
    answer:          str
