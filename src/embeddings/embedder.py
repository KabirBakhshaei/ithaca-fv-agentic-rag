"""
src/embeddings/embedder.py
==========================
Wraps a sentence-transformers model as a LangChain-compatible
Embeddings object so it can be plugged into ChromaDB seamlessly.

All embedding inference runs **locally** — no API call is made.
CUDA is used automatically when available (recommended on HPC H100s).
"""

from typing import List

from langchain_core.embeddings import Embeddings
from loguru import logger
from sentence_transformers import SentenceTransformer


class LocalEmbedder(Embeddings):
    """
    A LangChain Embeddings adapter around sentence-transformers.

    Usage
    -----
    embedder = LocalEmbedder(model_name="BAAI/bge-base-en-v1.5", device="cuda")
    vectorstore = Chroma(embedding_function=embedder, ...)
    """

    def __init__(self, model_name: str, device: str = "cuda", batch_size: int = 64):
        """
        Parameters
        ----------
        model_name:  HuggingFace model ID, e.g. "BAAI/bge-base-en-v1.5"
        device:      "cuda" or "cpu" — auto-falls back to cpu if no GPU
        batch_size:  number of texts encoded per forward pass
                     (increase to 256+ on H100 for maximum throughput)
        """
        logger.info(f"Loading embedding model '{model_name}' on device='{device}'…")
        try:
            self._model = SentenceTransformer(model_name, device=device)
        except Exception:
            # If CUDA is unavailable, fall back gracefully
            logger.warning(f"Could not load on '{device}', falling back to CPU.")
            self._model = SentenceTransformer(model_name, device="cpu")

        self._batch_size = batch_size
        self._model_name = model_name
        logger.success(f"Embedding model ready.")

    # ------------------------------------------------------------------
    # LangChain Embeddings interface
    # ------------------------------------------------------------------

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents (chunks).
        Called by the vector store during ingestion.
        """
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,   # needed for cosine similarity
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.
        Called by the vector store during retrieval.
        """
        # BGE models benefit from a query prefix
        prefix = (
            "Represent this sentence for searching relevant passages: "
            if "bge" in self._model_name.lower()
            else ""
        )
        vector = self._model.encode(
            prefix + text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector.tolist()
