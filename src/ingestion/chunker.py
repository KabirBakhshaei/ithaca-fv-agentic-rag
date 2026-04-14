"""
src/ingestion/chunker.py
========================
Splits raw LangChain Documents into smaller, overlapping chunks so that
the embedding model and context window are not overwhelmed.

Strategy
--------
* Markdown / RST files → split on headings first, then by character count.
* C++ (.H / .C) files  → split by blank lines (paragraph-style) to keep
  related declarations together.
* Python files          → split by class / function definitions when possible.
* Everything else       → plain recursive character splitter.

All chunks inherit the parent document's metadata plus a ``chunk_id`` field.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    Language,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)
from loguru import logger


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Split every document in *documents* into chunks.

    Parameters
    ----------
    documents:     raw documents from the loader
    chunk_size:    target characters per chunk
    chunk_overlap: characters shared between consecutive chunks

    Returns
    -------
    List[Document] — flattened list of all chunks across all documents
    """
    all_chunks: List[Document] = []

    for doc in documents:
        suffix = doc.metadata.get("suffix", "")
        splitter = _get_splitter(suffix, chunk_size, chunk_overlap)
        chunks = splitter.split_documents([doc])

        # Attach chunk index to metadata for traceability
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx
            chunk.metadata["total_chunks"] = len(chunks)

        all_chunks.extend(chunks)

    logger.info(
        f"Chunking complete: {len(documents)} docs → {len(all_chunks)} chunks "
        f"(avg {len(all_chunks) / max(len(documents), 1):.1f} chunks/doc)"
    )
    return all_chunks


# ---------------------------------------------------------------------------
# Internal: pick the right splitter per file type
# ---------------------------------------------------------------------------

def _get_splitter(
    suffix: str,
    chunk_size: int,
    chunk_overlap: int,
) -> RecursiveCharacterTextSplitter:
    """Return the most appropriate text splitter for a given file suffix."""

    if suffix in (".md", ".rst"):
        # Markdown-aware: tries to break on headings first
        return MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    if suffix == ".py":
        # Python-aware: tries to break on class / def boundaries
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    if suffix in (".H", ".C"):
        # C++ aware: tries to break on { } and blank lines
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.CPP,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # Default: generic recursive splitter
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
