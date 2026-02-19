#!/bin/bash
# GestureLSM BVH + ARKit Training Script for RunPod
#
# This script trains the full pipeline:
# 1. VQ-VAE for body (225D axis-angle)
# 2. VQ-VAE for face (51D ARKit blendshapes)
# 3. Shortcut generator (audio -> motion)
#
# Usage: bash run_training.sh
#
# Optimized for H100 (80GB VRAM) with PyTorch 2.1

set -e  # Exit on error

echo "=========================================="
echo "GestureLSM BVH + ARKit Training Pipeline"
echo "=========================================="

# =============================================================================
# Configuration - AUTO-SCALES for 1x to 8x H100
# =============================================================================
# Speakers: Use "$(seq 1 30)" for all 30 speakers
BEAT_SPEAKERS="$(seq 1 10)"

# Epochs - optimized for budget ($30 on 2x A100)
VQVAE_EPOCHS=300             # Good convergence, diminishing returns after
GENERATOR_EPOCHS=400         # Reasonable quality within budget

# Batch sizes - PER GPU (total = batch_size * num_gpus with DDP)
VQVAE_BATCH_SIZE=512         # Per GPU, A100 80GB can handle more
GENERATOR_BATCH_SIZE=192     # Per GPU, doubled for A100

# Paths
DATASET_PATH="./datasets/BEAT/beat_english_v0.2.1/beat_english_v0.2.1"
CACHE_PATH="./datasets/beat_cache/beat_bvh_arkit/"

# Workers - auto-scale based on vCPU count
VCPU_COUNT=$(nproc)
NUM_CACHE_WORKERS=$((VCPU_COUNT - 4))  # Leave some for system
NUM_DATA_WORKERS=$((VCPU_COUNT / 8))   # ~1 worker per 8 vCPUs
echo "Detected $VCPU_COUNT vCPUs: cache_workers=$NUM_CACHE_WORKERS, data_workers=$NUM_DATA_WORKERS"

# Multi-GPU settings
NUM_GPUS=$(nvidia-smi -L | wc -l)  # Auto-detect GPU count
USE_DDP=false
if [ "$NUM_GPUS" -gt 1 ]; then
    USE_DDP=true
    echo "Detected $NUM_GPUS GPUs - enabling DDP"
fi

# HuggingFace token for faster downloads (optional but recommended)
# Set HF_TOKEN env var before running: export HF_TOKEN="your_token_here"

# Step 0: Install dependencies
echo ""
echo "[Step 0] Installing dependencies..."
apt-get update && apt-get install -y git-lfs > /dev/null 2>&1 || true
git lfs install > /dev/null 2>&1 || true

# PyTorch 2.1 should already be installed on RunPod template
pip install -q omegaconf loguru wandb librosa smplx scipy tqdm matplotlib huggingface_hub pyyaml einops lmdb fasttext tensorboard transformers accelerate soundfile

# Step 1: Check/download dataset
echo ""
echo "[Step 1] Checking dataset..."

# Check if all speaker directories exist
MISSING_SPEAKERS=""
for speaker in $BEAT_SPEAKERS; do
    if [ ! -d "$DATASET_PATH/$speaker" ] || [ $(du -s "$DATASET_PATH/$speaker" 2>/dev/null | cut -f1) -lt 100000 ]; then
        MISSING_SPEAKERS="$MISSING_SPEAKERS $speaker"
    fi
done

if [ -n "$MISSING_SPEAKERS" ]; then
    echo "Missing or incomplete speaker directories:$MISSING_SPEAKERS"
    echo ""
    echo "Downloading from HuggingFace using huggingface_hub..."

    # Install huggingface_hub if needed
    pip install -q huggingface_hub

    # Download each missing speaker using huggingface_hub (more reliable than git lfs)
    for speaker in $MISSING_SPEAKERS; do
        echo "Downloading speaker $speaker..."
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='H-Liu1997/BEAT',
    repo_type='dataset',
    allow_patterns=['beat_english_v0.2.1/beat_english_v0.2.1/${speaker}/*'],
    local_dir='./datasets/BEAT',
    local_dir_use_symlinks=False
)
print('Speaker ${speaker} downloaded!')
"
    done

    echo "Dataset download complete!"
fi

# Verify at least one speaker exists
FOUND_SPEAKER=0
for speaker in $BEAT_SPEAKERS; do
    if [ -d "$DATASET_PATH/$speaker" ]; then
        FOUND_SPEAKER=1
        echo "Found speaker $speaker"
    fi
done

if [ $FOUND_SPEAKER -eq 0 ]; then
    echo "ERROR: No speaker data found at $DATASET_PATH"
    echo "Please download BEAT v0.2.1 and try again."
    exit 1
fi

echo "Dataset ready!"

# Step 1.5: Build cache FAST with parallel processing (skip if exists)
echo ""
echo "[Step 1.5] Checking cache..."

if [ -f "$CACHE_PATH/train_cache.pkl" ]; then
    echo "Cache already exists at $CACHE_PATH/train_cache.pkl - SKIPPING rebuild"
else
    echo "Building cache with parallel processing..."
    echo "Speakers: $BEAT_SPEAKERS"
    echo "Workers: $NUM_CACHE_WORKERS"

    python build_cache_fast.py \
        --data_path "$DATASET_PATH" \
        --cache_path "$CACHE_PATH" \
        --speakers $BEAT_SPEAKERS \
        --workers $NUM_CACHE_WORKERS

    if [ ! -f "$CACHE_PATH/train_cache.pkl" ]; then
        echo "ERROR: Cache building failed!"
        exit 1
    fi
    echo "Cache built successfully!"
fi

# Step 2: Train VQ-VAE for body
echo ""
echo "[Step 2] Training VQ-VAE for body (225D axis-angle)..."
echo "Speakers: $BEAT_SPEAKERS"
echo "Batch size: $VQVAE_BATCH_SIZE"
echo "Epochs: $VQVAE_EPOCHS"

if [ "$USE_DDP" = true ]; then
    torchrun --nproc_per_node=$NUM_GPUS train_vqvae_bvh.py \
        --body_part body \
        --epochs $VQVAE_EPOCHS \
        --batch_size $VQVAE_BATCH_SIZE \
        --speakers $BEAT_SPEAKERS \
        --data_path "$DATASET_PATH" \
        --cache_path "$CACHE_PATH" \
        --num_workers $NUM_DATA_WORKERS \
        --resume \
        --ddp
else
    python train_vqvae_bvh.py \
        --body_part body \
        --epochs $VQVAE_EPOCHS \
        --batch_size $VQVAE_BATCH_SIZE \
        --speakers $BEAT_SPEAKERS \
        --data_path "$DATASET_PATH" \
        --cache_path "$CACHE_PATH" \
        --num_workers $NUM_DATA_WORKERS \
        --resume
fi

# Check if VQ-VAE body training succeeded
if [ ! -f "./outputs/vqvae_bvh/body/best.pth" ]; then
    echo "ERROR: VQ-VAE body training failed!"
    exit 1
fi
echo "VQ-VAE body training complete!"

# Step 3: Train VQ-VAE for face
echo ""
echo "[Step 3] Training VQ-VAE for face (51D ARKit blendshapes)..."
if [ "$USE_DDP" = true ]; then
    torchrun --nproc_per_node=$NUM_GPUS train_vqvae_bvh.py \
        --body_part face \
        --epochs $VQVAE_EPOCHS \
        --batch_size $VQVAE_BATCH_SIZE \
        --speakers $BEAT_SPEAKERS \
        --data_path "$DATASET_PATH" \
        --cache_path "$CACHE_PATH" \
        --num_workers $NUM_DATA_WORKERS \
        --resume \
        --ddp
else
    python train_vqvae_bvh.py \
        --body_part face \
        --epochs $VQVAE_EPOCHS \
        --batch_size $VQVAE_BATCH_SIZE \
        --speakers $BEAT_SPEAKERS \
        --data_path "$DATASET_PATH" \
        --cache_path "$CACHE_PATH" \
        --num_workers $NUM_DATA_WORKERS \
        --resume
fi

# Check if VQ-VAE face training succeeded
if [ ! -f "./outputs/vqvae_bvh/face/best.pth" ]; then
    echo "ERROR: VQ-VAE face training failed!"
    exit 1
fi
echo "VQ-VAE face training complete!"

# Step 4: Update config and train shortcut generator
echo ""
echo "[Step 4] Training shortcut generator..."
echo "Batch size: $GENERATOR_BATCH_SIZE"
echo "Epochs: $GENERATOR_EPOCHS"

# Update config with correct settings
python3 << EOF
import yaml

with open('configs/shortcut_bvh_arkit.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# Update for H100 and 4 speakers
cfg['data']['new_cache'] = False  # Reuse existing cache from VQ-VAE training
cfg['data']['train_bs'] = $GENERATOR_BATCH_SIZE
cfg['data']['training_speakers'] = [1, 2, 3, 4]
cfg['data']['data_path'] = '$DATASET_PATH/'
cfg['training_speakers'] = [1, 2, 3, 4]
cfg['batch_size'] = $GENERATOR_BATCH_SIZE
cfg['solver']['epochs'] = $GENERATOR_EPOCHS

with open('configs/shortcut_bvh_arkit.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print('Updated config:')
print(f"  - Speakers: [1, 2, 3, 4]")
print(f"  - Batch size: $GENERATOR_BATCH_SIZE")
print(f"  - Epochs: $GENERATOR_EPOCHS")
EOF

# Find latest checkpoint for resume
LATEST_CKPT=""
if [ -d "./outputs/shortcut_bvh_arkit" ]; then
    LATEST_CKPT=$(ls -d ./outputs/shortcut_bvh_arkit/checkpoint_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
fi

if [ -n "$LATEST_CKPT" ] && [ -f "$LATEST_CKPT/ckpt.pth" ]; then
    echo "Resuming from checkpoint: $LATEST_CKPT"
    python train.py --config configs/shortcut_bvh_arkit.yaml --resume "$LATEST_CKPT"
else
    echo "Starting generator training from scratch"
    python train.py --config configs/shortcut_bvh_arkit.yaml
fi

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  VQ-VAE Body: ./outputs/vqvae_bvh/body/best.pth"
echo "  VQ-VAE Face: ./outputs/vqvae_bvh/face/best.pth"
echo "  Generator:   ./outputs/shortcut_bvh_arkit/best_fgd/ckpt.pth"
echo ""
echo "Normalization stats:"
echo "  Body mean: ./outputs/vqvae_bvh/body/body_mean.npy"
echo "  Body std:  ./outputs/vqvae_bvh/body/body_std.npy"
echo "  Face mean: ./outputs/vqvae_bvh/face/face_mean.npy"
echo "  Face std:  ./outputs/vqvae_bvh/face/face_std.npy"
echo ""
