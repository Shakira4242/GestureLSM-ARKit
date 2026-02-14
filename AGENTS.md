# GestureLSM - BVH + ARKit Training Plan

## What We're Doing
**Modifying GestureLSM** (in `/Users/akash/GestureLSM/`) to work with BVH + ARKit data.

We're NOT modifying DiffSHEG - we just borrowed their Euler→axis-angle conversion approach.

## Goal
Train GestureLSM (shortcut/fast model) on BEAT dataset with:
- **Body**: BVH format (75 joints × 3 axis-angle) = 225 dims
- **Face**: ARKit 51 blendshapes = 51 dims
- **Total**: 276 dims

## The Problem We Solved
Original training used raw BVH Euler angles which have **accumulated rotations** (e.g., -2482°).
This caused:
- Inconsistent training targets (same pose = different numbers)
- Huge normalization stats (mean=600°, std=2000°)
- Garbage output after denormalization

## The Fix
Convert **Euler angles → axis-angle** (borrowed from DiffSHEG):
- Axis-angle is bounded (roughly ±π radians)
- Each rotation has ONE representation
- Consistent training targets

## Key Files (all in GestureLSM/)

### Dataloaders
| File | Purpose |
|------|---------|
| `dataloaders/beat_normalized.py` | BVH + ARKit loader with Euler→axis-angle conversion |
| `dataloaders/rotation_converter.py` | Euler↔axis-angle conversion (copied from DiffSHEG) |

### Training Scripts
| File | Purpose |
|------|---------|
| `train_vqvae_bvh.py` | Train VQ-VAE for body (225D) and face (51D) |
| `train.py` | Train shortcut generator (use with BVH config) |

### Models
| File | Purpose |
|------|---------|
| `models/vq/model.py` | RVQVAE - unchanged, just different input dims |
| `models/LSM.py` | Shortcut generator - unchanged |

### Configs
| File | Purpose |
|------|---------|
| `configs/shortcut_bvh_arkit.yaml` | Main config for BVH+ARKit generator training |
| `configs/sc_model_bvh_config.yaml` | Model architecture config for BVH |

## Training Pipeline

```
Step 1: Train VQ-VAEs
─────────────────────
python train_vqvae_bvh.py --body_part body --epochs 100
python train_vqvae_bvh.py --body_part face --epochs 100

# This creates:
#   ./outputs/vqvae_bvh/body/best.pth + body_mean.npy + body_std.npy
#   ./outputs/vqvae_bvh/face/best.pth + face_mean.npy + face_std.npy

Step 2: Train Shortcut Generator
────────────────────────────────
python train.py --config configs/shortcut_bvh_arkit.yaml

# This creates:
#   ./outputs/shortcut_bvh_arkit/best.pth

Step 3: Inference (TODO)
────────────────────────
python inference_bvh.py --audio input.wav --output motion.npy

Step 4: Stream to Unity (TODO)
──────────────────────────────
python stream_to_unity.py --motion motion.npy
```

## Data Format

### Input (BEAT v0.2.1)
```
datasets/BEAT/beat_english_v0.2.1/beat_english_v0.2.1/
├── 2/                          # Speaker ID
│   ├── 2_scott_0_1_1.bvh      # Body motion (BVH Euler)
│   ├── 2_scott_0_1_1.json     # Face (ARKit 51 blendshapes)
│   └── 2_scott_0_1_1.wav      # Audio
```

### After Processing
```
Body: 75 joints × 3 axis-angle (radians) = 225 dims
Face: 51 ARKit blendshapes = 51 dims
Total: 276 dims per frame
```

### Normalization Stats (expected ranges)
```
Body axis-angle: mean ≈ 0, std ≈ 0.5-2.0 (radians)
Face blendshapes: mean ≈ 0.1-0.3, std ≈ 0.1-0.3 (0-1 range)
```

## Unity Streaming

### Output Format
```
47 joints × 4 quaternion floats = 188 floats (body)
51 ARKit blendshapes = 51 floats (face)
Total: 239 floats per UDP packet
```

### Conversion Pipeline
```
axis-angle (225D) → select 47 joints → quaternion (188 floats)
ARKit (51D) → pass through → (51 floats)
```

## References
- DiffSHEG: https://github.com/dunbar12138/DiffSHEG (reference for axis-angle approach)
- GestureLSM: https://github.com/andypinxinliu/GestureLSM (original uses SMPL-X, we're adapting for BVH)
- BEAT Dataset: https://pantomatrix.github.io/BEAT/

## TODO
- [x] Copy `rotation_converter.py` from DiffSHEG to GestureLSM
- [x] Update `beat_normalized.py` to convert Euler → axis-angle
- [x] Create `configs/shortcut_bvh_arkit.yaml` for generator training
- [x] Create `configs/sc_model_bvh_config.yaml` for model architecture
- [x] Update `trainer/base_trainer.py` for BVH mode (skip SMPL-X)
- [x] Update `trainer/generative_trainer.py` for BVH mode (BVH VQ models, BVH validation)
- [x] Create `run_training.sh` for RunPod
- [ ] Train VQ-VAE (body + face) on RunPod
- [ ] Train shortcut generator on RunPod
- [ ] Create inference script
- [ ] Update Unity streaming (axis-angle → quaternion)
- [ ] Test full pipeline

## RunPod Training

```bash
# Clone and run
git clone <your-repo>
cd GestureLSM
bash run_training.sh
```

This will:
1. Install dependencies
2. Download sample data if needed
3. Train VQ-VAE for body (225D)
4. Train VQ-VAE for face (51D)
5. Train shortcut generator

## MISTAKES TO AVOID
- **DO NOT create new training scripts or models from scratch** (e.g., `train_all_bvh.py`, `simple_generator.py`)
- **USE the existing GestureLSM architecture** - just adapt configs and dataloaders
- The existing `train.py`, `train_vqvae_bvh.py`, and `trainer/generative_trainer.py` should be used/adapted, not replaced
