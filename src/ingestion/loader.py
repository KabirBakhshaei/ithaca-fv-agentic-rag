"""
src/ingestion/loader.py
=======================
Clones the ITHACA-FV GitHub repository (or uses an existing local copy)
and loads all supported documents into LangChain Document objects.

Supported file types (configurable in config.yaml):
  .md / .rst  → Markdown / reStructuredText documentation
  .H  / .C    → OpenFOAM C++ headers and source files
  .py         → Python tutorials and test scripts
  .txt        → Plain-text notes
"""

import os
import subprocess
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from loguru import logger
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helper: clone or update the repo
# ---------------------------------------------------------------------------

def _clone_or_update_repo(repo_url: str, local_path: str) -> None:
    """
    If *local_path* does not exist, clone *repo_url* there.
    If it already exists, perform a `git pull` to update it.
    """
    local = Path(local_path)

    if local.exists():
        logger.info(f"Repo already exists at {local_path}. Running git pull…")
        result = subprocess.run(
            ["git", "-C", str(local), "pull"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            logger.warning(f"git pull failed: {result.stderr.strip()}")
        else:
            logger.info(result.stdout.strip())
    else:
        logger.info(f"Cloning {repo_url} → {local_path}…")
        local.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(local)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"git clone failed: {result.stderr.strip()}")
        logger.success(f"Cloned successfully.")


# ---------------------------------------------------------------------------
# Helper: walk files
# ---------------------------------------------------------------------------

def _walk_files(
    root: Path,
    extensions: List[str],
    exclude_dirs: List[str],
) -> List[Path]:
    """Recursively yield all files matching *extensions* below *root*."""
    found: List[Path] = []
    for path in root.rglob("*"):
        # Skip excluded directories
        if any(ex in path.parts for ex in exclude_dirs):
            continue
        if path.is_file() and path.suffix in extensions:
            found.append(path)
    return found


# ---------------------------------------------------------------------------
# Main loader function
# ---------------------------------------------------------------------------

def load_ithaca_fv_documents(
    repo_url: str,
    local_repo_path: str,
    file_extensions: List[str],
    exclude_dirs: List[str],
) -> List[Document]:
    """
    1. Clone / update the ITHACA-FV repository.
    2. Walk all files matching *file_extensions*.
    3. Read each file and wrap it in a LangChain Document with rich metadata.

    Returns
    -------
    List[Document]
        Each document carries the file content as ``page_content`` and
        metadata fields: ``source``, ``file_type``, ``relative_path``,
        ``repo``.
    """
    # Step 1 — ensure repo is available locally
    _clone_or_update_repo(repo_url, local_repo_path)

    repo_root = Path(local_repo_path)

    # Step 2 — discover files
    files = _walk_files(repo_root, file_extensions, exclude_dirs)
    logger.info(f"Found {len(files)} files to ingest.")

    # Step 3 — read and wrap
    documents: List[Document] = []
    for fpath in tqdm(files, desc="Loading files", unit="file"):
        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            logger.warning(f"Skipping {fpath}: {exc}")
            continue

        # Skip empty files
        if not content.strip():
            continue

        relative = fpath.relative_to(repo_root)

        # Infer a human-readable doc type for metadata
        doc_type = _infer_doc_type(fpath)

        doc = Document(
            page_content=content,
            metadata={
                "source":        str(fpath),           # absolute path
                "relative_path": str(relative),        # path inside repo
                "file_name":     fpath.name,
                "file_type":     doc_type,
                "repo":          "ITHACA-FV",
                "suffix":        fpath.suffix,
            },
        )
        documents.append(doc)

    logger.success(f"Loaded {len(documents)} documents from ITHACA-FV repo.")
    return documents


def _infer_doc_type(path: Path) -> str:
    """Map file extension to a human-readable category."""
    mapping = {
        ".md":  "markdown_doc",
        ".rst": "rst_doc",
        ".H":   "cpp_header",
        ".C":   "cpp_source",
        ".py":  "python_script",
        ".txt": "text_doc",
    }
    return mapping.get(path.suffix, "other")
