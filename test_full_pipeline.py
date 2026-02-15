#!/usr/bin/env python3
"""
Full Pipeline Sanity Check - Tests audio → generator → VQ-VAE → motion

This is the CRITICAL test to catch jittery motion before you waste time.

Usage:
    python test_full_pipeline.py                    # Basic test
    python test_full_pipeline.py --num_samples 10   # More samples
    python test_full_pipeline.py --save_output      # Save motion for visualization
"""

import os
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


def compute_velocity(motion):
    """Compute frame-to-frame velocity."""
    if len(motion.shape) == 2:
        return motion[1:] - motion[:-1]
    else:
        return motion[:, 1:] - motion[:, :-1]


def load_vqvae(checkpoint_path, dim_pose, device='cuda'):
    """Load a trained VQ-VAE model."""
    from models.vq.model import RVQVAE

    class Args:
        num_quantizers = 6
        shared_codebook = False
        quantize_dropout_prob = 0.2
        quantize_dropout_cutoff_index = 0
        mu = 0.99

    model = RVQVAE(
        Args(),
        dim_pose,
        nb_code=1024,
        code_dim=128,
        output_emb_width=128,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation='relu',
        norm=None
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['net'])
    model.eval()
    return model


def load_generator(config_path, checkpoint_path, device='cuda'):
    """Load the generator model."""
    import yaml
    from omegaconf import OmegaConf
    from models.config import instantiate_from_config

    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg)

    # Instantiate model
    model_cfg = cfg.model
    denoiser = instantiate_from_config(model_cfg.denoiser)
    modality_encoder = instantiate_from_config(model_cfg.modality_encoder)

    # Build full LSM model
    from models.LSM import GestureLSM
    model = GestureLSM(
        denoiser=denoiser,
        modality_encoder=modality_encoder,
        n_steps=model_cfg.n_steps,
        do_classifier_free_guidance=model_cfg.do_classifier_free_guidance,
        guidance_scale=model_cfg.guidance_scale,
    ).to(device)

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        elif 'net' in ckpt:
            model.load_state_dict(ckpt['net'])
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded generator checkpoint")
    else:
        print(f"WARNING: Generator checkpoint not found: {checkpoint_path}")
        return None

    model.eval()
    return model, cfg


def run_inference(model, vq_body, vq_face, audio_mel, seed_motion, cfg, device='cuda'):
    """Run generator inference on audio."""
    with torch.no_grad():
        bs = audio_mel.shape[0]
        n_frames = audio_mel.shape[1]

        # Prepare conditioning
        cond_ = {"y": {}}
        cond_["y"]["audio_onset"] = audio_mel
        cond_["y"]["word"] = None
        cond_["y"]["id"] = None
        cond_["y"]["seed"] = seed_motion
        cond_["y"]["style_feature"] = None

        # Generate latents
        output = model(cond_)
        latents = output["latents"]
        latents = latents.squeeze(2).permute(0, 2, 1)

        # Split body and face
        code_dim = vq_body.code_dim
        latent_body = latents[..., :code_dim] * cfg.vqvae_latent_scale
        latent_face = latents[..., code_dim:code_dim*2] * cfg.vqvae_latent_scale

        # Decode through VQ-VAE
        rec_body = vq_body.latent2origin(latent_body)[0]
        rec_face = vq_face.latent2origin(latent_face)[0]

        return rec_body, rec_face


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--save_output', action='store_true')
    parser.add_argument('--config', type=str, default='./configs/shortcut_bvh_arkit.yaml')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("\n" + "="*60)
    print("FULL PIPELINE SANITY CHECK")
    print("Audio → Generator → VQ-VAE → Motion → Jitter Check")
    print("="*60)

    # Check required files
    required = {
        'VQ-VAE Body': './outputs/vqvae_bvh/body/best.pth',
        'VQ-VAE Face': './outputs/vqvae_bvh/face/best.pth',
        'Generator': './outputs/shortcut_bvh_arkit/best_fgd/ckpt.pth',
        'Body Mean': './outputs/vqvae_bvh/body/body_mean.npy',
        'Body Std': './outputs/vqvae_bvh/body/body_std.npy',
        'Face Mean': './outputs/vqvae_bvh/face/face_mean.npy',
        'Face Std': './outputs/vqvae_bvh/face/face_std.npy',
        'Cache': './datasets/beat_cache/beat_bvh_arkit/train_cache.pkl',
    }

    # Also check for epoch checkpoints if best_fgd doesn't exist
    gen_ckpt = required['Generator']
    if not os.path.exists(gen_ckpt):
        import glob
        alt_ckpts = glob.glob('./outputs/shortcut_bvh_arkit/**/*.pth', recursive=True)
        if alt_ckpts:
            gen_ckpt = sorted(alt_ckpts)[-1]  # Use latest
            required['Generator'] = gen_ckpt
            print(f"Using alternative checkpoint: {gen_ckpt}")

    missing = []
    for name, path in required.items():
        if not os.path.exists(path):
            missing.append(f"  - {name}: {path}")

    if missing:
        print("\n[ERROR] Missing required files:")
        for m in missing:
            print(m)
        print("\nGenerator training may not have completed yet.")
        print("Run this after generator training finishes.")
        return False

    print("\n[OK] All required files found!")

    # Load models
    print("\nLoading models...")

    vq_body = load_vqvae('./outputs/vqvae_bvh/body/best.pth', 225, device)
    vq_face = load_vqvae('./outputs/vqvae_bvh/face/best.pth', 51, device)
    print("  VQ-VAE Body: loaded")
    print("  VQ-VAE Face: loaded")

    # Load generator
    try:
        generator, cfg = load_generator(args.config, gen_ckpt, device)
        if generator is None:
            return False
        print("  Generator: loaded")
    except Exception as e:
        print(f"\n[ERROR] Failed to load generator: {e}")
        print("\nThis might happen if generator training hasn't saved a checkpoint yet.")
        return False

    # Load normalization stats
    body_mean = torch.from_numpy(np.load('./outputs/vqvae_bvh/body/body_mean.npy')).float().to(device)
    body_std = torch.from_numpy(np.load('./outputs/vqvae_bvh/body/body_std.npy')).float().to(device)
    face_mean = torch.from_numpy(np.load('./outputs/vqvae_bvh/face/face_mean.npy')).float().to(device)
    face_std = torch.from_numpy(np.load('./outputs/vqvae_bvh/face/face_std.npy')).float().to(device)

    # Load dataset cache
    print("\nLoading dataset...")
    with open('./datasets/beat_cache/beat_bvh_arkit/train_cache.pkl', 'rb') as f:
        cache = pickle.load(f)

    samples = cache['samples'][:args.num_samples]
    print(f"  Testing on {len(samples)} samples")

    # Run inference and check jitter
    print("\n" + "-"*60)
    print("RUNNING FULL PIPELINE INFERENCE")
    print("-"*60)

    all_body_vel_ratio = []
    all_face_vel_ratio = []
    all_body_mse = []
    all_face_mse = []
    all_body_spike = []
    all_face_spike = []

    for i, sample in enumerate(samples):
        # Get ground truth
        gt_body = sample['body']  # (T, 225)
        gt_face = sample['face']  # (T, 51)
        mel = sample['mel']  # (T, 128)

        # Normalize ground truth (for seed)
        gt_body_norm = (gt_body - cache['body_mean']) / cache['body_std']

        # Prepare inputs
        mel_t = torch.from_numpy(mel).float().unsqueeze(0).to(device)  # (1, T, 128)

        # Use first few frames as seed
        n_seed = 8
        seed = torch.from_numpy(gt_body_norm[:n_seed]).float().unsqueeze(0).to(device)

        # Encode seed through VQ-VAE to get latent seed
        with torch.no_grad():
            seed_full = torch.from_numpy(gt_body_norm).float().unsqueeze(0).to(device)
            _, seed_latent = vq_body.encode(seed_full)
            seed_latent = seed_latent.mean(dim=0)  # Average over quantizers
            seed_latent = seed_latent[:, :n_seed, :]  # First n_seed frames

        try:
            # Run inference
            rec_body, rec_face = run_inference(
                generator, vq_body, vq_face,
                mel_t, seed_latent, cfg, device
            )

            # Denormalize
            rec_body = rec_body.cpu().numpy()[0]
            rec_face = rec_face.cpu().numpy()[0]
            rec_body = rec_body * cache['body_std'] + cache['body_mean']
            rec_face = rec_face * cache['face_std'] + cache['face_mean']

            # Align lengths
            min_len = min(len(gt_body), len(rec_body))
            gt_body = gt_body[:min_len]
            gt_face = gt_face[:min_len]
            rec_body = rec_body[:min_len]
            rec_face = rec_face[:min_len]

            # Compute metrics
            # MSE
            body_mse = np.mean((gt_body - rec_body) ** 2)
            face_mse = np.mean((gt_face - rec_face) ** 2)

            # Velocity (jitter check)
            vel_gt_body = compute_velocity(gt_body)
            vel_rec_body = compute_velocity(rec_body)
            vel_gt_face = compute_velocity(gt_face)
            vel_rec_face = compute_velocity(rec_face)

            body_vel_ratio = np.std(vel_rec_body) / (np.std(vel_gt_body) + 1e-7)
            face_vel_ratio = np.std(vel_rec_face) / (np.std(vel_gt_face) + 1e-7)

            # Spike detection
            body_spike = np.max(np.abs(vel_rec_body)) / (np.max(np.abs(vel_gt_body)) + 1e-7)
            face_spike = np.max(np.abs(vel_rec_face)) / (np.max(np.abs(vel_gt_face)) + 1e-7)

            all_body_mse.append(body_mse)
            all_face_mse.append(face_mse)
            all_body_vel_ratio.append(body_vel_ratio)
            all_face_vel_ratio.append(face_vel_ratio)
            all_body_spike.append(body_spike)
            all_face_spike.append(face_spike)

            status = "[OK]" if body_vel_ratio < 1.5 else "[WARN]" if body_vel_ratio < 2.5 else "[BAD]"
            print(f"  Sample {i+1}: body_vel={body_vel_ratio:.2f} {status}, face_vel={face_vel_ratio:.2f}")

            # Save output if requested
            if args.save_output:
                os.makedirs('./outputs/sanity_check', exist_ok=True)
                np.save(f'./outputs/sanity_check/sample_{i}_rec_body.npy', rec_body)
                np.save(f'./outputs/sanity_check/sample_{i}_rec_face.npy', rec_face)
                np.save(f'./outputs/sanity_check/sample_{i}_gt_body.npy', gt_body)
                np.save(f'./outputs/sanity_check/sample_{i}_gt_face.npy', gt_face)

        except Exception as e:
            print(f"  Sample {i+1}: ERROR - {e}")
            continue

    if not all_body_vel_ratio:
        print("\n[ERROR] No samples processed successfully!")
        return False

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    avg_body_vel = np.mean(all_body_vel_ratio)
    avg_face_vel = np.mean(all_face_vel_ratio)
    avg_body_mse = np.mean(all_body_mse)
    avg_face_mse = np.mean(all_face_mse)
    max_body_spike = np.max(all_body_spike)
    max_face_spike = np.max(all_face_spike)

    print(f"\nBODY MOTION:")
    print(f"  MSE:           {avg_body_mse:.6f}")
    print(f"  Velocity ratio: {avg_body_vel:.3f}  {'[OK]' if avg_body_vel < 1.5 else '[BAD - JITTERY!]' if avg_body_vel > 2.0 else '[WARN]'}")
    print(f"  Max spike:      {max_body_spike:.3f}  {'[OK]' if max_body_spike < 3 else '[BAD]' if max_body_spike > 5 else '[WARN]'}")

    print(f"\nFACE MOTION:")
    print(f"  MSE:           {avg_face_mse:.6f}")
    print(f"  Velocity ratio: {avg_face_vel:.3f}  {'[OK]' if avg_face_vel < 1.5 else '[BAD - JITTERY!]' if avg_face_vel > 2.0 else '[WARN]'}")
    print(f"  Max spike:      {max_face_spike:.3f}  {'[OK]' if max_face_spike < 3 else '[BAD]' if max_face_spike > 5 else '[WARN]'}")

    # Final verdict
    print("\n" + "="*60)
    print("FINAL VERDICT:")
    print("="*60)

    has_jitter = avg_body_vel > 2.0 or avg_face_vel > 2.0
    has_spikes = max_body_spike > 5 or max_face_spike > 5

    if has_jitter:
        print("\n[FAIL] JITTERY MOTION DETECTED!")
        print("       This is the problem you had before.")
        print("       Velocity ratio > 2.0 means frame-to-frame jumps are too large.")
        print("\n       Possible causes:")
        print("       1. Generator not trained long enough")
        print("       2. Generator not learning audio correlation")
        print("       3. VQ-VAE latent mismatch")
        return False
    elif has_spikes:
        print("\n[WARN] Some frame spikes detected")
        print("       Motion has occasional large jumps but may be acceptable.")
        return True
    else:
        print("\n[PASS] Motion looks smooth!")
        print("       Velocity ratios are healthy.")
        print("       No significant jitter detected.")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
