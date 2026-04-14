#!/bin/bash
#SBATCH --job-name=ithaca_rag_serve
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4                # use all 4 × H100 for vLLM (tensor parallel)
#SBATCH --mem=64G
#SBATCH --time=08:00:00             # long job — keeps the GUI alive for a day
#SBATCH --output=logs/serve_%j.out
#SBATCH --error=logs/serve_%j.err

# ============================================================
#  ITHACA-FV RAG — Serving Job  (vLLM + Gradio)
#
#  This script:
#    1. Starts a vLLM OpenAI-compatible server on port 8000
#       (using all 4 × H100 in tensor-parallel mode)
#    2. Waits for the server to be ready
#    3. Launches the Gradio app on port 7860
#
#  After submitting, SSH-tunnel to your laptop:
#    ssh -L 7860:hpcsrv:7860 k.bakhshaei@hpcsrv
#  Then open: http://localhost:7860
#
#  OR use --share=True in config.yaml for a public Gradio URL.
#
#  Submit with:  sbatch scripts/slurm/serve.slurm
# ============================================================

set -euo pipefail

PROJECT_DIR="/storage/home/k.bakhshaei/ithaca_fv_rag"
MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"   # or local HuggingFace cache path
VLLM_PORT=8000
GRADIO_PORT=7860

cd "$PROJECT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pinns_cuda

mkdir -p logs

echo "=============================="
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Node        : $SLURMD_NODENAME"
echo "  GPUs        : $CUDA_VISIBLE_DEVICES"
echo "  Start time  : $(date)"
echo "=============================="

# ── Step 1: Start vLLM server in the background ──────────────────────────────
echo "[1/2] Starting vLLM server (tensor_parallel=4)…"
vllm serve "$MODEL_PATH" \
    --tensor-parallel-size 4 \
    --port $VLLM_PORT \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    &
VLLM_PID=$!

# ── Wait for vLLM to be ready ─────────────────────────────────────────────────
echo "Waiting for vLLM to start…"
for i in $(seq 1 60); do
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo "vLLM is ready after ${i}s."
        break
    fi
    sleep 5
done

# ── Step 2: Launch Gradio app ─────────────────────────────────────────────────
echo "[2/2] Launching Gradio app on port $GRADIO_PORT…"
echo "  → SSH tunnel:  ssh -L ${GRADIO_PORT}:${SLURMD_NODENAME}:${GRADIO_PORT} k.bakhshaei@hpcsrv"
python scripts/run_app.py --port $GRADIO_PORT

# ── Cleanup on exit ───────────────────────────────────────────────────────────
kill $VLLM_PID 2>/dev/null || true
echo "Serve job finished at $(date)"
