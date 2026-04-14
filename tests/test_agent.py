"""
tests/test_agent.py
===================
Unit and integration tests for the agentic RAG pipeline.

Run with:
    pytest tests/ -v

For a quick smoke-test without a GPU or LLM:
    pytest tests/ -v -k "not integration"
"""

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document

from src.rag.state import AgentState


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_docs() -> List[Document]:
    return [
        Document(
            page_content="ITHACA-FV is an implementation of reduced order modelling "
                         "techniques in OpenFOAM.",
            metadata={"source": "README.md", "relative_path": "README.md",
                      "file_type": "markdown_doc"},
        ),
        Document(
            page_content="To install ITHACA-FV, first install OpenFOAM v9, "
                         "then clone the repository and run Allwmake.",
            metadata={"source": "docs/install.md", "relative_path": "docs/install.md",
                      "file_type": "markdown_doc"},
        ),
    ]


@pytest.fixture
def base_state(sample_docs) -> AgentState:
    return AgentState(
        question        = "How do I install ITHACA-FV?",
        query_type      = "",
        search_query    = "How do I install ITHACA-FV?",
        needs_retrieval = True,
        documents       = [],
        relevant_docs   = [],
        is_sufficient   = False,
        retry_count     = 0,
        answer          = "",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chunker tests  (no GPU / LLM required)
# ─────────────────────────────────────────────────────────────────────────────

class TestChunker:
    def test_markdown_chunking(self, sample_docs):
        from src.ingestion.chunker import chunk_documents

        chunks = chunk_documents(sample_docs, chunk_size=200, chunk_overlap=20)
        assert len(chunks) >= len(sample_docs), "Should produce at least as many chunks as docs"

    def test_chunk_metadata_preserved(self, sample_docs):
        from src.ingestion.chunker import chunk_documents

        chunks = chunk_documents(sample_docs, chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert "source" in chunk.metadata, "source metadata must be preserved"
            assert "chunk_id" in chunk.metadata, "chunk_id must be added"

    def test_empty_input(self):
        from src.ingestion.chunker import chunk_documents

        result = chunk_documents([], chunk_size=500, chunk_overlap=50)
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# Node unit tests  (LLM is mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestNodes:

    def _make_llm_mock(self, response_text: str):
        """Return a mock LLM that returns *response_text* for any invoke call."""
        mock = MagicMock()
        mock.invoke.return_value = MagicMock(content=response_text)
        return mock

    def test_query_analyzer_yes_retrieval(self, base_state):
        from src.rag.nodes import query_analyzer

        llm = self._make_llm_mock(
            "CATEGORY: installation\nNEEDS_RETRIEVAL: YES\nSEARCH_QUERY: ITHACA-FV install OpenFOAM"
        )
        result = query_analyzer(base_state, llm=llm)

        assert result["query_type"] == "installation"
        assert result["needs_retrieval"] is True
        assert "ITHACA-FV" in result["search_query"]

    def test_query_analyzer_no_retrieval(self, base_state):
        from src.rag.nodes import query_analyzer

        llm = self._make_llm_mock(
            "CATEGORY: general\nNEEDS_RETRIEVAL: NO\nSEARCH_QUERY: What is C++"
        )
        state = {**base_state, "question": "What is C++?"}
        result = query_analyzer(state, llm=llm)

        assert result["needs_retrieval"] is False

    def test_relevance_grader_filters(self, base_state, sample_docs):
        from src.rag.nodes import relevance_grader

        # First doc is relevant (YES), second is not (NO)
        call_count = [0]
        def side_effect(prompt):
            call_count[0] += 1
            return MagicMock(content="YES" if call_count[0] == 1 else "NO")

        llm = MagicMock()
        llm.invoke.side_effect = side_effect

        state = {**base_state, "documents": sample_docs}
        result = relevance_grader(state, llm=llm)

        assert len(result["relevant_docs"]) == 1
        assert result["is_sufficient"] is True

    def test_relevance_grader_empty(self, base_state):
        from src.rag.nodes import relevance_grader

        llm = self._make_llm_mock("NO")
        state = {**base_state, "documents": []}
        result = relevance_grader(state, llm=llm)

        assert result["relevant_docs"] == []
        assert result["is_sufficient"] is False

    def test_query_rewriter_increments_retry(self, base_state):
        from src.rag.nodes import query_rewriter

        llm = self._make_llm_mock("ITHACA-FV compilation Allwmake OpenFOAM")
        state = {**base_state, "retry_count": 0}
        result = query_rewriter(state, llm=llm)

        assert result["retry_count"] == 1
        assert result["search_query"] != base_state["search_query"]

    def test_generator_with_context(self, base_state, sample_docs):
        from src.rag.nodes import generator

        llm = self._make_llm_mock(
            "To install ITHACA-FV: 1) Install OpenFOAM, 2) Clone the repo, 3) Run Allwmake."
        )
        state = {**base_state, "relevant_docs": sample_docs}
        result = generator(state, llm=llm)

        assert len(result["answer"]) > 0

    def test_generator_no_context(self, base_state):
        from src.rag.nodes import generator

        llm = self._make_llm_mock("I could not find specific documentation…")
        state = {**base_state, "relevant_docs": []}
        result = generator(state, llm=llm)

        assert len(result["answer"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Graph routing tests  (no LLM)
# ─────────────────────────────────────────────────────────────────────────────

class TestRouting:

    def test_route_to_retriever_when_needed(self, base_state):
        from src.rag.agent import _route_after_analyzer

        state = {**base_state, "needs_retrieval": True}
        assert _route_after_analyzer(state) == "retriever"

    def test_route_to_generator_when_not_needed(self, base_state):
        from src.rag.agent import _route_after_analyzer

        state = {**base_state, "needs_retrieval": False}
        assert _route_after_analyzer(state) == "generator"

    def test_route_to_generator_when_sufficient(self, base_state):
        from src.rag.agent import _route_after_grader

        state = {**base_state, "is_sufficient": True, "retry_count": 0}
        assert _route_after_grader(state, max_retries=2) == "generator"

    def test_route_to_rewriter_when_not_sufficient_and_retries_left(self, base_state):
        from src.rag.agent import _route_after_grader

        state = {**base_state, "is_sufficient": False, "retry_count": 0}
        assert _route_after_grader(state, max_retries=2) == "query_rewriter"

    def test_route_to_generator_when_retries_exhausted(self, base_state):
        from src.rag.agent import _route_after_grader

        state = {**base_state, "is_sufficient": False, "retry_count": 2}
        assert _route_after_grader(state, max_retries=2) == "generator"


# ─────────────────────────────────────────────────────────────────────────────
# Integration test  (requires local Ollama — mark with -k integration to run)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestEndToEnd:
    """
    Requires:
    - Ollama running locally: ollama serve
    - ChromaDB already built: python scripts/ingest.py
    - At least one GPU for embeddings
    """

    def test_full_pipeline_install_question(self):
        import yaml
        from src.rag.agent import load_agent

        with open("config/config.yaml") as f:
            config = yaml.safe_load(f)

        agent = load_agent(config)
        result = agent.invoke({
            "question":        "How do I install ITHACA-FV?",
            "query_type":      "",
            "search_query":    "How do I install ITHACA-FV?",
            "needs_retrieval": True,
            "documents":       [],
            "relevant_docs":   [],
            "is_sufficient":   False,
            "retry_count":     0,
            "answer":          "",
        })

        assert "answer" in result
        assert len(result["answer"]) > 50, "Answer should be non-trivial"
        print(f"\n[Integration] Answer:\n{result['answer']}\n")
