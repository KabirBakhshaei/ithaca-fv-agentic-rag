from setuptools import find_packages, setup

setup(
    name="ithaca_fv_rag",
    version="0.1.0",
    description="Agentic RAG assistant for the ITHACA-FV reduced-order modelling library",
    author="Kabir Bakhshaei",
    url="https://github.com/YOUR_USERNAME/ithaca-fv-rag",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "langchain>=0.2.0",
        "langchain-community>=0.2.0",
        "langchain-core>=0.2.0",
        "langgraph>=0.1.0",
        "langchain-openai>=0.1.0",
        "sentence-transformers>=3.0.0",
        "torch>=2.2.0",
        "chromadb>=0.5.0",
        "gitpython>=3.1.40",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "gradio>=4.30.0",
        "tqdm>=4.66.0",
        "loguru>=0.7.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0", "pytest-asyncio>=0.23.0"],
        "ollama": ["langchain-ollama"],
        "vllm": ["vllm"],
    },
)
