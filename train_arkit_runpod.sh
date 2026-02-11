#!/bin/bash
# ============================================================================
# GestureLSM ARKit Training Script for RunPod
# ============================================================================
# GPU: RTX 4090 (~$0.44/hr) or A100 (~$1.10/hr)
# Estimated time: ~20-25 hours total
# Estimated cost: ~$10-15 on RTX 4090
#
# What this trains:
# 1. Face VQ-VAE for ARKit 51 blendshapes (~3-4 hrs)
# 2. MeanFlow generator with face support (~15-20 hrs)
# ============================================================================

set -e

echo "=============================================="
echo "GestureLSM ARKit Training Setup"
echo "=============================================="
echo ""

# Check GPU
nvidia-smi || echo "Warning: No GPU detected"

# ============================================================================
# STEP 1: Setup Environment
# ============================================================================
echo "[1/7] Setting up environment..."

cd /workspace

# Clone repo if not exists
if [ ! -d "GestureLSM" ]; then
    git clone https://github.com/PantoMatrix/GestureLSM
fi
cd GestureLSM

# Install dependencies
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q -r requirements.txt
pip install -q huggingface_hub gdown loguru

# ============================================================================
# STEP 2: Download Pretrained Models (Body VQ-VAEs)
# ============================================================================
echo "[2/7] Downloading pretrained body VQ-VAEs..."

mkdir -p ckpt
gdown https://drive.google.com/drive/folders/1OfYWWJbaXal6q7LttQlYKWAy0KTwkPRw -O ./ckpt --folder

# ============================================================================
# STEP 3: Download SMPL Models
# ============================================================================
echo "[3/7] Downloading SMPL models..."

mkdir -p datasets/hub
gdown https://drive.google.com/drive/folders/1MCks7CMNBtAzU2XihYezNmiGT_6pWex8 -O ./datasets/hub --folder

# ============================================================================
# STEP 4: Download BEAT Dataset (Speaker 2 only)
# ============================================================================
echo "[4/7] Downloading BEAT dataset..."

# Original BEAT (has ARKit JSON face data)
mkdir -p datasets/BEAT
huggingface-cli download H-Liu1997/BEAT \
    --local-dir ./datasets/BEAT \
    --include "beat_english_v0.2.1/beat_english_v0.2.1/2/*"

# BEAT2 (has SMPL-X body NPZ)
mkdir -p datasets/BEAT2
huggingface-cli download H-Liu1997/BEAT2 \
    --local-dir ./datasets/BEAT2 \
    --include "beat_english_v2.0.0/smplxflame_30/2_*"

echo "Dataset download complete!"
echo "  BEAT JSON (face): $(find datasets/BEAT -name '*.json' | wc -l) files"
echo "  BEAT NPZ (body): $(find datasets/BEAT2 -name '*.npz' | wc -l) files"

# ============================================================================
# STEP 5: Copy Custom Training Files
# ============================================================================
echo "[5/7] Setting up custom training files..."

# The custom files should already be in the repo, but if running from scratch:
# - meanflow_trainer.py
# - configs/meanflow_arkit.yaml
# - configs/mf_arkit_model_config.yaml
# - dataloaders/beat_arkit.py

# ============================================================================
# STEP 6: Train Face VQ-VAE (51-dim ARKit)
# ============================================================================
echo "[6/7] Training Face VQ-VAE for ARKit..."
echo "      This will take ~3-4 hours..."

python train_face_vq_arkit.py \
    --epochs 300 \
    --batch_size 128 \
    --lr 1e-4 \
    --save_dir ./ckpt

echo "Face VQ-VAE training complete!"

# ============================================================================
# STEP 7: Train MeanFlow Generator
# ============================================================================
echo "[7/7] Training MeanFlow generator with ARKit face..."
echo "      This will take ~15-20 hours..."

python meanflow_trainer.py \
    -c configs/meanflow_arkit.yaml \
    --epochs 500 \
    --lr 1e-4 \
    --batch_size 128 \
    --save_dir ./outputs/meanflow_arkit

echo "=============================================="
echo "TRAINING COMPLETE!"
echo "=============================================="
echo ""
echo "Output files:"
echo "  - Face VQ: ./ckpt/face_arkit_51.pth"
echo "  - MeanFlow: ./outputs/meanflow_arkit/meanflow_epoch500.pth"
echo ""
echo "To download to your local machine:"
echo "  scp -r runpod:/workspace/GestureLSM/ckpt/face_arkit_51.pth ."
echo "  scp -r runpod:/workspace/GestureLSM/outputs/meanflow_arkit/*.pth ."
echo ""
