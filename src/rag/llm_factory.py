"""
src/rag/llm_factory.py
======================
Single factory function that returns a LangChain chat model based on
the ``llm`` section of ``config/config.yaml``.

Supported providers
-------------------
  "ollama"     → Ollama running locally (no API key needed)
                 Install: https://ollama.ai
                 Then:    ollama pull llama3.1:8b

  "vllm"       → vLLM server (OpenAI-compatible, great for H100s)
                 Recommended for HPC: supports tensor parallelism across GPUs
                 Install: pip install vllm
                 Run:     vllm serve meta-llama/Llama-3.1-8B-Instruct \
                              --tensor-parallel-size 4

  "openrouter" → Free cloud API (good fallback if no local GPU available)
                 Sign up: https://openrouter.ai  (no credit card needed)
                 Set:     OPENROUTER_API_KEY in .env
                 Free models include: meta-llama/llama-3.1-8b-instruct:free
                                      mistralai/mistral-7b-instruct:free
"""

import os

from langchain_core.language_models import BaseChatModel
from loguru import logger


def get_llm(llm_cfg: dict) -> BaseChatModel:
    """
    Instantiate and return the correct LangChain chat model.

    Parameters
    ----------
    llm_cfg: the ``llm`` section of config.yaml as a Python dict

    Returns
    -------
    BaseChatModel — ready to call with ``.invoke(prompt)``
    """
    provider    = llm_cfg["provider"].lower()
    model       = llm_cfg["model"]
    temperature = llm_cfg.get("temperature", 0.1)
    max_tokens  = llm_cfg.get("max_tokens", 1024)

    # ------------------------------------------------------------------ #
    #  OLLAMA  — fully local, no API key required                         #
    # ------------------------------------------------------------------ #
    if provider == "ollama":
        from langchain_ollama import ChatOllama  # pip install langchain-ollama

        base_url = llm_cfg.get("base_url", "http://localhost:11434")
        logger.info(f"[LLM] Using Ollama | model={model} | url={base_url}")
        return ChatOllama(
            model=model,
            temperature=temperature,
            num_predict=max_tokens,
            base_url=base_url,
        )

    # ------------------------------------------------------------------ #
    #  vLLM  — OpenAI-compatible local server, best for H100 cluster     #
    # ------------------------------------------------------------------ #
    if provider == "vllm":
        from langchain_openai import ChatOpenAI

        base_url = llm_cfg.get("base_url", "http://localhost:8000/v1")
        logger.info(f"[LLM] Using vLLM | model={model} | url={base_url}")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key="not-needed",         # vLLM does not require a key
        )

    # ------------------------------------------------------------------ #
    #  OPENROUTER  — free cloud API, good fallback                        #
    # ------------------------------------------------------------------ #
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY not set. "
                "Copy .env.example to .env and add your key."
            )
        base_url = llm_cfg.get("base_url", "https://openrouter.ai/api/v1")
        logger.info(f"[LLM] Using OpenRouter | model={model}")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key,
        )

    raise ValueError(
        f"Unknown LLM provider '{provider}'. "
        "Choose from: ollama, vllm, openrouter"
    )
