#!/bin/bash
#SBATCH --job-name=ithaca_rag_ingest
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # sentence-transformers uses multiple CPU cores
#SBATCH --gres=gpu:1                # 1 × H100 is plenty for embedding
#SBATCH --mem=32G
#SBATCH --time=02:00:00             # ingestion usually takes 20–40 min
#SBATCH --output=logs/ingest_%j.out
#SBATCH --error=logs/ingest_%j.err

# ============================================================
#  ITHACA-FV RAG — Ingestion Job
#  Submit with:  sbatch scripts/slurm/ingest.slurm
# ============================================================

set -euo pipefail

echo "=============================="
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Node        : $SLURMD_NODENAME"
echo "  GPUs        : $CUDA_VISIBLE_DEVICES"
echo "  Start time  : $(date)"
echo "=============================="

# --- Project root (edit this to your actual path) ---
PROJECT_DIR="/storage/home/k.bakhshaei/ithaca_fv_rag"
cd "$PROJECT_DIR"

# --- Activate conda environment ---
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pinns_cuda      # reuse your existing env, or create a new one

# --- Create logs directory ---
mkdir -p logs

# --- Run ingestion ---
python scripts/ingest.py --config config/config.yaml

echo "=============================="
echo "  Ingestion finished : $(date)"
echo "=============================="
