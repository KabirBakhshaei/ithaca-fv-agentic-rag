"""
src/vectorstore/chroma_store.py
================================
Thin wrapper around ChromaDB that provides two entry points:

1. ``build_vectorstore``  — ingest chunks and persist to disk (run once)
2. ``load_vectorstore``   — load an existing on-disk collection (used at query time)

ChromaDB stores all data locally under ``config.vectorstore.persist_dir``.
No internet connection is required after the initial ingestion.
"""

from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from loguru import logger


# ---------------------------------------------------------------------------
# Build (run once during ingestion)
# ---------------------------------------------------------------------------

def build_vectorstore(
    chunks: List[Document],
    embedder: Embeddings,
    persist_dir: str,
    collection_name: str,
) -> Chroma:
    """
    Embed all *chunks* and persist them to a ChromaDB collection on disk.

    Parameters
    ----------
    chunks:          list of text chunks produced by the chunker
    embedder:        LocalEmbedder (or any LangChain Embeddings)
    persist_dir:     directory where ChromaDB files will be saved
    collection_name: logical name for this collection

    Returns
    -------
    Chroma  — the populated vector store object
    """
    logger.info(
        f"Building ChromaDB collection '{collection_name}' "
        f"with {len(chunks)} chunks…"
    )

    # Chroma.from_documents automatically batches embedding calls
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        collection_name=collection_name,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},  # use cosine distance
    )

    logger.success(
        f"Vector store built and persisted at '{persist_dir}'. "
        f"Total vectors: {vectorstore._collection.count()}"
    )
    return vectorstore


# ---------------------------------------------------------------------------
# Load (used at inference time)
# ---------------------------------------------------------------------------

def load_vectorstore(
    embedder: Embeddings,
    persist_dir: str,
    collection_name: str,
) -> Chroma:
    """
    Load an already-built ChromaDB collection from disk.
    Call this in the RAG agent instead of re-ingesting every time.

    Raises
    ------
    FileNotFoundError if *persist_dir* does not exist.
    """
    import os
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(
            f"No ChromaDB found at '{persist_dir}'. "
            "Run `python scripts/ingest.py` first."
        )

    logger.info(f"Loading ChromaDB collection '{collection_name}' from '{persist_dir}'…")
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedder,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )
    count = vectorstore._collection.count()
    logger.success(f"Loaded {count} vectors from ChromaDB.")
    return vectorstore


# ---------------------------------------------------------------------------
# Retriever factory (convenience)
# ---------------------------------------------------------------------------

def get_retriever(
    vectorstore: Chroma,
    top_k: int = 5,
    score_threshold: Optional[float] = 0.35,
) -> VectorStoreRetriever:
    """
    Return a LangChain retriever that filters results below *score_threshold*.

    Parameters
    ----------
    vectorstore:      the loaded/built Chroma instance
    top_k:            maximum number of documents to return per query
    score_threshold:  minimum cosine similarity (0–1); lower = more docs
    """
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": top_k,
            "score_threshold": score_threshold,
        },
    )
