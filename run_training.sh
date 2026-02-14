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

# Configuration - optimized for H100
BEAT_SPEAKERS="1 2 3 4"      # Train on 4 speakers
VQVAE_EPOCHS=100
GENERATOR_EPOCHS=500
VQVAE_BATCH_SIZE=256         # H100 can handle larger batches
GENERATOR_BATCH_SIZE=256     # H100 80GB VRAM
DATASET_PATH="./datasets/BEAT/beat_english_v0.2.1/beat_english_v0.2.1"

# Step 0: Install dependencies
echo ""
echo "[Step 0] Installing dependencies..."
# PyTorch 2.1 should already be installed on RunPod template
pip install -q omegaconf loguru wandb librosa smplx scipy tqdm matplotlib huggingface_hub pyyaml

# Step 1: Check/download dataset
echo ""
echo "[Step 1] Checking dataset..."

# Check if all speaker directories exist
MISSING_SPEAKERS=""
for speaker in $BEAT_SPEAKERS; do
    if [ ! -d "$DATASET_PATH/$speaker" ]; then
        MISSING_SPEAKERS="$MISSING_SPEAKERS $speaker"
    fi
done

if [ -n "$MISSING_SPEAKERS" ]; then
    echo "Missing speaker directories:$MISSING_SPEAKERS"
    echo ""
    echo "Attempting to download from HuggingFace..."

    python3 << EOF
import os
from huggingface_hub import snapshot_download

dataset_path = "$DATASET_PATH"
speakers = [int(s) for s in "$BEAT_SPEAKERS".split()]

try:
    # Download the dataset from HuggingFace
    print("Downloading BEAT dataset from H-Liu1997/BEAT...")
    print("This may take a while...")

    # Download entire dataset first
    snapshot_download(
        repo_id="H-Liu1997/BEAT",
        repo_type="dataset",
        local_dir="./datasets/BEAT_download/",
    )

    # Move speaker folders to correct location
    os.makedirs(dataset_path, exist_ok=True)
    for speaker in speakers:
        src = f"./datasets/BEAT_download/beat_english_v0.2.1/{speaker}"
        dst = os.path.join(dataset_path, str(speaker))
        if os.path.exists(src) and not os.path.exists(dst):
            os.system(f"mv '{src}' '{dst}'")
            print(f"Speaker {speaker} ready!")
        elif os.path.exists(dst):
            print(f"Speaker {speaker} already exists")
        else:
            print(f"Speaker {speaker} not found in download")

    print("Dataset download complete!")

except Exception as e:
    print(f"Download failed: {e}")
    print("")
    print("Please download BEAT manually from:")
    print("  https://huggingface.co/datasets/H-Liu1997/BEAT")
    print("")
    print("Extract to: $DATASET_PATH")
EOF
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

# Step 2: Train VQ-VAE for body
echo ""
echo "[Step 2] Training VQ-VAE for body (225D axis-angle)..."
echo "Speakers: $BEAT_SPEAKERS"
echo "Batch size: $VQVAE_BATCH_SIZE"
echo "Epochs: $VQVAE_EPOCHS"

python train_vqvae_bvh.py \
    --body_part body \
    --epochs $VQVAE_EPOCHS \
    --batch_size $VQVAE_BATCH_SIZE \
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
    --batch_size $VQVAE_BATCH_SIZE \
    --speakers $BEAT_SPEAKERS \
    --data_path "$DATASET_PATH"

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
cfg['data']['new_cache'] = True
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
