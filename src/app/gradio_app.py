"""
src/app/gradio_app.py
=====================
Gradio-based chat interface for the ITHACA-FV Agentic RAG assistant.

Features
--------
* Chat history preserved within a session
* Source documents expandable in an accordion
* Question category badge (installation / api / tutorial / ...)
* "Clear" button to reset the conversation
* PDF upload  -> text extracted and added to ChromaDB (permanent ingestion)
* Image upload -> sent directly to Gemma 4 vision (multimodal)
* Works on HPC via share=True (Gradio tunnel) or via SSH port forwarding

Run with:
    python scripts/run_app.py
Or directly:
    python -m src.app.gradio_app
"""

import base64
import os
import time

import gradio as gr
import yaml
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


# ---------------------------------------------------------------------------
# Config & agent loader
# ---------------------------------------------------------------------------

def _load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_agent_from_config(config: dict):
    from src.rag.agent import load_agent
    return load_agent(config)


# ---------------------------------------------------------------------------
# PDF helpers — extract text and ingest into ChromaDB permanently
# ---------------------------------------------------------------------------

def _extract_pdf_text(pdf_path: str) -> str:
    """Extract all text from a PDF. Tries PyMuPDF first, then pdfminer."""
    try:
        import fitz  # pip install pymupdf
        doc = fitz.open(pdf_path)
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except ImportError:
        pass
    try:
        from pdfminer.high_level import extract_text  # pip install pdfminer.six
        return extract_text(pdf_path).strip()
    except ImportError:
        raise ImportError(
            "No PDF library found. Install one:\n"
            "  pip install pymupdf        (recommended)\n"
            "  pip install pdfminer.six   (fallback)"
        )


def _ingest_pdf_into_chroma(pdf_path, pdf_name, vectorstore, embedder, chunk_size=1000, chunk_overlap=150):
    """
    Extract text from pdf_path, chunk it, embed it, and add it to the
    live ChromaDB collection so future queries can retrieve from it.
    Returns the number of chunks added.
    """
    from langchain_core.documents import Document
    from src.ingestion.chunker import chunk_documents

    text = _extract_pdf_text(pdf_path)
    if not text:
        logger.warning(f"[PDF] No text extracted from '{pdf_name}'.")
        return 0

    doc = Document(
        page_content=text,
        metadata={
            "source":        pdf_path,
            "relative_path": pdf_name,
            "file_name":     pdf_name,
            "file_type":     "user_uploaded_pdf",
            "suffix":        ".pdf",
        },
    )
    chunks = chunk_documents([doc], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vectorstore.add_documents(chunks)
    logger.success(f"[PDF] '{pdf_name}' ingested — {len(chunks)} chunks added to ChromaDB.")
    return len(chunks)


# ---------------------------------------------------------------------------
# Image helpers — encode to base64 for Gemma 4 vision
# ---------------------------------------------------------------------------

def _encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_media_type(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif",
            ".webp": "image/webp"}.get(ext, "image/jpeg")


# ---------------------------------------------------------------------------
# Core chat function
# ---------------------------------------------------------------------------

def chat(user_message, history, uploaded_pdf, uploaded_image,
         agent, config, vectorstore, embedder):
    """
    Handle one turn of the conversation.

    Parameters
    ----------
    user_message   : text typed by the user
    history        : list of {"role":..., "content":...} dicts (Gradio 6)
    uploaded_pdf   : file path string from gr.File, or None
    uploaded_image : file path string from gr.Image, or None
    agent          : compiled LangGraph agent
    config         : loaded YAML config dict
    vectorstore    : live ChromaDB instance (needed for PDF ingestion)
    embedder       : LocalEmbedder instance (needed for PDF ingestion)

    Returns
    -------
    (history, sources_md, meta_label, pdf_status)
    """
    if not user_message.strip() and uploaded_image is None:
        return history, "", "", ""

    start      = time.time()
    pdf_status = ""
    ing_cfg    = config.get("ingestion", {})

    # ── 1. PDF upload: extract and ingest into ChromaDB ───────────────────
    if uploaded_pdf is not None:
        pdf_name = os.path.basename(uploaded_pdf)
        try:
            n = _ingest_pdf_into_chroma(
                pdf_path=uploaded_pdf,
                pdf_name=pdf_name,
                vectorstore=vectorstore,
                embedder=embedder,
                chunk_size=ing_cfg.get("chunk_size", 1000),
                chunk_overlap=ing_cfg.get("chunk_overlap", 150),
            )
            pdf_status = (
                f"✅ **'{pdf_name}'** ingested — {n} chunks added to the "
                f"knowledge base. Retrieval will now include this document."
            )
        except Exception as exc:
            pdf_status = f"⚠️ PDF ingestion failed: {exc}"
            logger.error(f"[PDF] {exc}")

    # ── 2. Build question (with optional image context hint) ──────────────
    question = user_message.strip() or "Describe the image and relate it to ITHACA-FV."

    if uploaded_image is not None:
        question += (
            "\n\n[An image has been attached. Please describe what you see, "
            "then answer the question in the context of ITHACA-FV and "
            "reduced-order modelling if relevant.]"
        )
        logger.info(f"[Image] Attached: {os.path.basename(uploaded_image)}")

    # ── 3. Run the agentic RAG workflow ───────────────────────────────────
    result = agent.invoke({
        "question":        question,
        "query_type":      "",
        "search_query":    user_message or "image analysis ITHACA-FV",
        "needs_retrieval": True,
        "documents":       [],
        "relevant_docs":   [],
        "is_sufficient":   False,
        "retry_count":     0,
        "answer":          "",
    })

    elapsed  = time.time() - start
    answer   = result.get("answer", "Sorry, I could not generate an answer.")
    category = result.get("query_type", "general").capitalize()
    sources_md = _format_sources(result.get("relevant_docs", []))

    # ── 4. Build history entry (show image thumbnail in chat if uploaded) ──
    if uploaded_image is not None:
        user_display = {
            "role": "user",
            "content": (
                f"![uploaded]({uploaded_image})\n\n{user_message}"
                if user_message.strip()
                else f"![uploaded]({uploaded_image})"
            ),
        }
    else:
        user_display = {"role": "user", "content": user_message}

    history = history + [
        user_display,
        {"role": "assistant", "content": answer},
    ]

    logger.info(f"Answered in {elapsed:.1f}s | category={category}")
    return (
        history,
        sources_md,
        f"🏷️  Category: **{category}**  ⏱️  {elapsed:.1f}s",
        pdf_status,
    )


def _format_sources(docs) -> str:
    if not docs:
        return "_No source documents retrieved for this question._"
    parts = []
    for i, doc in enumerate(docs, start=1):
        meta    = doc.metadata
        source  = meta.get("relative_path", meta.get("source", "unknown"))
        ftype   = meta.get("file_type", "")
        snippet = doc.page_content[:400].replace("\n", "\n> ")
        parts.append(f"**[{i}] `{source}`** `{ftype}`\n\n> {snippet}...\n")
    return "\n---\n".join(parts)


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def build_ui(agent, config, vectorstore, embedder) -> gr.Blocks:
    app_cfg = config.get("app", {})
    title   = app_cfg.get("title", "ITHACA-FV Assistant")
    desc    = app_cfg.get(
        "description",
        "Ask anything about the ITHACA-FV reduced-order modelling library."
    )

    with gr.Blocks(title=title) as demo:

        # ── Header ──────────────────────────────────────────────────────
        gr.Markdown(f"""
# 🔬 {title}
**{desc}**

Powered by [ITHACA-FV](https://github.com/ITHACA-FV/ITHACA-FV) ·
[LangGraph](https://github.com/langchain-ai/langgraph) ·
Local LLM + ChromaDB · 📄 PDF & 🖼️ Image support
        """)

        # ── Chat window ──────────────────────────────────────────────────
        chatbot = gr.Chatbot(
            label="Conversation",
            elem_classes=["chat-window"],
        )

        # ── Text input ───────────────────────────────────────────────────
        with gr.Row():
            msg_box = gr.Textbox(
                placeholder="e.g. How do I install ITHACA-FV on Ubuntu?",
                label="Your question",
                lines=2,
                scale=8,
            )
            submit_btn = gr.Button("Ask", variant="primary", scale=1)

        # ── File upload row ──────────────────────────────────────────────
        with gr.Row():
            pdf_upload = gr.File(
                label="📄 Upload PDF  —  ingested into knowledge base permanently",
                file_types=[".pdf"],
                scale=1,
            )
            image_upload = gr.Image(
                label="🖼️ Upload Image  —  analysed by Gemma 4 vision",
                type="filepath",
                scale=1,
            )

        # ── PDF ingestion status ─────────────────────────────────────────
        pdf_status_box = gr.Markdown("", label="")

        # ── Category / timing badge ──────────────────────────────────────
        meta_label = gr.Markdown("", label="")

        # ── Retrieved sources accordion ──────────────────────────────────
        with gr.Accordion("📄 Retrieved Source Documents", open=False):
            sources_box = gr.Markdown(
                "_Ask a question to see source documents._",
                elem_classes=["source-box"],
            )

        # ── Example questions ────────────────────────────────────────────
        gr.Examples(
            examples=[
                "How do I install and compile ITHACA-FV?",
                "What is the ITHACASolver class used for?",
                "How do I run the cavity tutorial in ITHACA-FV?",
                "Explain the POD basis construction in ITHACA-FV.",
                "What OpenFOAM versions are compatible with ITHACA-FV?",
                "How does DEIM work in the context of ITHACA-FV?",
                "Upload an error screenshot and ask: What does this error mean?",
            ],
            inputs=msg_box,
            label="💡 Example Questions",
        )

        # ── Clear button ─────────────────────────────────────────────────
        with gr.Row():
            clear_btn = gr.Button("🗑️  Clear conversation", variant="secondary")

        # ── Event handlers ───────────────────────────────────────────────
        def _submit(user_msg, history, pdf_file, img_file):
            return chat(
                user_message=user_msg,
                history=history,
                uploaded_pdf=pdf_file,
                uploaded_image=img_file,
                agent=agent,
                config=config,
                vectorstore=vectorstore,
                embedder=embedder,
            )

        def _clear():
            return [], "", "", "", None, None

        submit_btn.click(
            fn=_submit,
            inputs=[msg_box, chatbot, pdf_upload, image_upload],
            outputs=[chatbot, sources_box, meta_label, pdf_status_box],
        )
        msg_box.submit(
            fn=_submit,
            inputs=[msg_box, chatbot, pdf_upload, image_upload],
            outputs=[chatbot, sources_box, meta_label, pdf_status_box],
        )
        clear_btn.click(
            fn=_clear,
            inputs=[],
            outputs=[chatbot, sources_box, meta_label, pdf_status_box,
                     pdf_upload, image_upload],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def launch(config_path: str = "config/config.yaml"):
    from src.embeddings.embedder import LocalEmbedder
    from src.vectorstore.chroma_store import load_vectorstore

    config = _load_config(config_path)
    logger.info("Initialising ITHACA-FV RAG agent...")

    # Keep embedder and vectorstore alive so PDF ingestion works at runtime
    emb_cfg  = config["embeddings"]
    embedder = LocalEmbedder(
        model_name=emb_cfg["model"],
        device=emb_cfg.get("device", "cuda"),
        batch_size=emb_cfg.get("batch_size", 64),
    )
    vs_cfg      = config["vectorstore"]
    vectorstore = load_vectorstore(
        embedder=embedder,
        persist_dir=vs_cfg["persist_dir"],
        collection_name=vs_cfg["collection_name"],
    )

    agent   = _build_agent_from_config(config)
    app_cfg = config.get("app", {})
    demo    = build_ui(agent, config, vectorstore, embedder)

    demo.launch(
        server_name=app_cfg.get("host", "0.0.0.0"),
        server_port=app_cfg.get("port", 7860),
        share=app_cfg.get("share", False),
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css=(
            ".chat-window { height: 520px !important; } "
            ".source-box  { font-size: 0.85rem; } "
            "footer { display: none !important; }"
        ),
    )


if __name__ == "__main__":
    launch()
