#!/bin/bash
# GestureLSM BVH + ARKit Training Script for RunPod
#
# This script trains the full pipeline:
# 1. VQ-VAE for body (225D axis-angle)
# 2. VQ-VAE for face (51D ARKit blendshapes)
# 3. Shortcut generator (audio -> motion)
#
# Usage: bash run_training.sh

set -e  # Exit on error

echo "=========================================="
echo "GestureLSM BVH + ARKit Training Pipeline"
echo "=========================================="

# Configuration
BEAT_SPEAKERS="2"  # Which speakers to train on (comma-separated, e.g., "1,2,3")
VQVAE_EPOCHS=100
GENERATOR_EPOCHS=500
BATCH_SIZE=128

# Step 0: Install dependencies
echo ""
echo "[Step 0] Installing dependencies..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q omegaconf loguru wandb librosa smplx scipy tqdm matplotlib

# Step 1: Check/download dataset
echo ""
echo "[Step 1] Checking dataset..."
DATASET_PATH="./datasets/BEAT/beat_english_v0.2.1/beat_english_v0.2.1"

if [ ! -d "$DATASET_PATH/2" ]; then
    echo "Dataset not found. Please download BEAT v0.2.1 from:"
    echo "  https://pantomatrix.github.io/BEAT/"
    echo ""
    echo "And extract to: $DATASET_PATH"
    echo ""
    echo "Directory structure should be:"
    echo "  $DATASET_PATH/2/2_scott_0_1_1.bvh"
    echo "  $DATASET_PATH/2/2_scott_0_1_1.json"
    echo "  $DATASET_PATH/2/2_scott_0_1_1.wav"
    echo ""

    # Try to download sample data from HuggingFace
    echo "Attempting to download sample data from HuggingFace..."
    mkdir -p "$DATASET_PATH/2"

    pip install -q huggingface_hub
    python3 << 'EOF'
import os
from huggingface_hub import hf_hub_download

dataset_path = "./datasets/BEAT/beat_english_v0.2.1/beat_english_v0.2.1/2"

try:
    # Download one sample file for testing
    for ext in ['.bvh', '.json', '.wav']:
        filename = f"2_scott_0_1_1{ext}"
        try:
            path = hf_hub_download(
                repo_id="beat2/beat_v2.0.0_backup",
                filename=f"beat_english_v2.0.0/2/{filename}",
                repo_type="dataset",
                local_dir="./datasets/temp/"
            )
            os.makedirs(dataset_path, exist_ok=True)
            os.system(f"cp '{path}' '{dataset_path}/{filename}'")
            print(f"Downloaded: {filename}")
        except Exception as e:
            print(f"Could not download {filename}: {e}")
    print("Sample data downloaded!")
except Exception as e:
    print(f"Could not download from HuggingFace: {e}")
    print("Please download the dataset manually.")
EOF
fi

# Verify dataset exists
if [ ! -f "$DATASET_PATH/2/2_scott_0_1_1.bvh" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    echo "Please download BEAT v0.2.1 and try again."
    exit 1
fi

echo "Dataset found!"

# Step 2: Train VQ-VAE for body
echo ""
echo "[Step 2] Training VQ-VAE for body (225D axis-angle)..."
python train_vqvae_bvh.py \
    --body_part body \
    --epochs $VQVAE_EPOCHS \
    --batch_size $BATCH_SIZE \
    --speakers $BEAT_SPEAKERS \
    --data_path "$DATASET_PATH" \
    --new_cache

# Check if VQ-VAE body training succeeded
if [ ! -f "./outputs/vqvae_bvh/body/best.pth" ]; then
    echo "ERROR: VQ-VAE body training failed!"
    exit 1
fi
echo "VQ-VAE body training complete!"

# Step 3: Train VQ-VAE for face
echo ""
echo "[Step 3] Training VQ-VAE for face (51D ARKit blendshapes)..."
python train_vqvae_bvh.py \
    --body_part face \
    --epochs $VQVAE_EPOCHS \
    --batch_size $BATCH_SIZE \
    --speakers $BEAT_SPEAKERS \
    --data_path "$DATASET_PATH"

# Check if VQ-VAE face training succeeded
if [ ! -f "./outputs/vqvae_bvh/face/best.pth" ]; then
    echo "ERROR: VQ-VAE face training failed!"
    exit 1
fi
echo "VQ-VAE face training complete!"

# Step 4: Train shortcut generator
echo ""
echo "[Step 4] Training shortcut generator..."

# First, rebuild cache with new_cache=True to ensure fresh data
python -c "
import yaml
with open('configs/shortcut_bvh_arkit.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['data']['new_cache'] = True
with open('configs/shortcut_bvh_arkit.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print('Updated config for fresh cache')
"

python train.py --config configs/shortcut_bvh_arkit.yaml

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
