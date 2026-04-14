"""
scripts/ingest.py
=================
One-time script to clone the ITHACA-FV repository, chunk all documents,
embed them, and persist to ChromaDB.

Run this BEFORE launching the chat app:

    python scripts/ingest.py [--config config/config.yaml]

On HPC, submit via SLURM:

    sbatch scripts/slurm/ingest.slurm
"""

import argparse
import sys
import time
from pathlib import Path

# Make sure the project root is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

load_dotenv()
console = Console()


def main(config_path: str = "config/config.yaml") -> None:
    # ------------------------------------------------------------------ #
    #  1. Load configuration                                              #
    # ------------------------------------------------------------------ #
    with open(config_path) as f:
        config = yaml.safe_load(f)

    console.print(Panel.fit(
        "[bold blue]ITHACA-FV Agentic RAG — Document Ingestion[/bold blue]\n"
        f"Config: {config_path}",
        border_style="blue",
    ))

    # ------------------------------------------------------------------ #
    #  2. Clone / update the ITHACA-FV repository                        #
    # ------------------------------------------------------------------ #
    from src.ingestion.loader import load_ithaca_fv_documents

    ing_cfg   = config["ingestion"]
    t0        = time.time()

    logger.info("Step 1/4 — Loading documents from repository…")
    documents = load_ithaca_fv_documents(
        repo_url       = ing_cfg["repo_url"],
        local_repo_path= ing_cfg["local_repo_path"],
        file_extensions= ing_cfg["file_extensions"],
        exclude_dirs   = ing_cfg["exclude_dirs"],
    )
    console.print(f"  ✔  Loaded [bold]{len(documents)}[/bold] raw documents  "
                  f"({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------ #
    #  3. Chunk documents                                                  #
    # ------------------------------------------------------------------ #
    from src.ingestion.chunker import chunk_documents

    logger.info("Step 2/4 — Chunking documents…")
    t1     = time.time()
    chunks = chunk_documents(
        documents=documents,
        chunk_size   =ing_cfg["chunk_size"],
        chunk_overlap=ing_cfg["chunk_overlap"],
    )
    console.print(f"  ✔  Created [bold]{len(chunks)}[/bold] chunks  "
                  f"({time.time()-t1:.1f}s)")

    # ------------------------------------------------------------------ #
    #  4. Load embedding model                                            #
    # ------------------------------------------------------------------ #
    from src.embeddings.embedder import LocalEmbedder

    emb_cfg = config["embeddings"]
    logger.info(f"Step 3/4 — Loading embedding model '{emb_cfg['model']}'…")
    t2      = time.time()
    embedder = LocalEmbedder(
        model_name=emb_cfg["model"],
        device    =emb_cfg.get("device", "cuda"),
        batch_size=emb_cfg.get("batch_size", 64),
    )
    console.print(f"  ✔  Embedding model ready  ({time.time()-t2:.1f}s)")

    # ------------------------------------------------------------------ #
    #  5. Build and persist ChromaDB vector store                         #
    # ------------------------------------------------------------------ #
    from src.vectorstore.chroma_store import build_vectorstore

    vs_cfg = config["vectorstore"]
    logger.info("Step 4/4 — Building ChromaDB vector store…")
    t3     = time.time()
    build_vectorstore(
        chunks         =chunks,
        embedder       =embedder,
        persist_dir    =vs_cfg["persist_dir"],
        collection_name=vs_cfg["collection_name"],
    )
    console.print(f"  ✔  Vector store persisted at [bold]{vs_cfg['persist_dir']}[/bold]  "
                  f"({time.time()-t3:.1f}s)")

    # ------------------------------------------------------------------ #
    #  Done                                                               #
    # ------------------------------------------------------------------ #
    total = time.time() - t0
    console.print(Panel.fit(
        f"[bold green]Ingestion complete![/bold green]\n"
        f"Total time: {total:.0f}s\n\n"
        f"You can now run the chat app:\n"
        f"  [cyan]python scripts/run_app.py[/cyan]",
        border_style="green",
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest ITHACA-FV docs into ChromaDB")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    main(args.config)
