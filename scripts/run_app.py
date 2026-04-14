"""
scripts/run_app.py
==================
Launch the Gradio chat interface.

Local usage:
    python scripts/run_app.py

HPC usage (get a public URL via Gradio tunnel):
    python scripts/run_app.py --share

HPC usage with SSH tunnel (no public URL needed):
    # On HPC node, run:
    python scripts/run_app.py
    # On your laptop, run:
    ssh -L 7860:localhost:7860 k.bakhshaei@hpcsrv
    # Then open:  http://localhost:7860
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Launch ITHACA-FV RAG chat app")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio URL (useful on HPC with no open ports)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override port from config (default: 7860)",
    )
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides config
    if args.share:
        config["app"]["share"] = True
    if args.port:
        config["app"]["port"] = args.port

    from src.app.gradio_app import launch
    launch(args.config)


if __name__ == "__main__":
    main()
