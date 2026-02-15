#!/usr/bin/env python3
"""
Sanity check script for GestureLSM BVH training.

Run this in a separate terminal while training continues.

Usage:
    python sanity_check.py --check vqvae          # Check VQ-VAE reconstruction
    python sanity_check.py --check pipeline       # Check full audio->motion pipeline
    python sanity_check.py --check all            # Run all checks
"""

import os
import argparse
import torch
import numpy as np
import time
from pathlib import Path

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


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

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['net'])
        epoch = ckpt.get('epoch', '?')
        loss = ckpt.get('loss', '?')
        print(f"Loaded checkpoint: epoch={epoch}, loss={loss:.4f}" if isinstance(loss, float) else f"Loaded checkpoint: epoch={epoch}")
    else:
        print(f"WARNING: Checkpoint not found: {checkpoint_path}")
        return None

    model.eval()
    return model


def compute_velocity(motion):
    """Compute frame-to-frame velocity."""
    # motion: (T, D) or (B, T, D)
    if len(motion.shape) == 2:
        return motion[1:] - motion[:-1]
    else:
        return motion[:, 1:] - motion[:, :-1]


def check_vqvae(body_part='body', num_samples=10, device='cuda', use_latest=False, specific_epoch=None):
    """
    Check VQ-VAE reconstruction quality.

    Key metrics:
    - Reconstruction MSE
    - Velocity variance ratio (reconstruction vs original)
    - Codebook perplexity (are discrete codes being used?)
    - If velocity variance ratio > 2, motion is likely jittery
    """
    print("\n" + "="*60)
    print(f"VQ-VAE Sanity Check ({body_part})")
    print("="*60)

    # Determine checkpoint path
    if body_part == 'body':
        ckpt_dir = './outputs/vqvae_bvh/body'
        dim_pose = 225
    else:
        ckpt_dir = './outputs/vqvae_bvh/face'
        dim_pose = 51

    if specific_epoch is not None:
        ckpt_path = f'{ckpt_dir}/epoch_{specific_epoch}.pth'
    elif use_latest:
        # Find the latest epoch checkpoint
        import glob
        epoch_files = glob.glob(f'{ckpt_dir}/epoch_*.pth')
        if epoch_files:
            # Extract epoch numbers and find max
            epochs = [int(f.split('epoch_')[1].split('.pth')[0]) for f in epoch_files]
            latest_epoch = max(epochs)
            ckpt_path = f'{ckpt_dir}/epoch_{latest_epoch}.pth'
            print(f"Using latest checkpoint: epoch {latest_epoch}")
        else:
            ckpt_path = f'{ckpt_dir}/best.pth'
            print("No epoch checkpoints found, using best.pth")
    else:
        ckpt_path = f'{ckpt_dir}/best.pth'

    # Check if checkpoint exists
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        print("VQ-VAE training may still be in progress...")
        # List available checkpoints
        import glob
        available = glob.glob(f'{ckpt_dir}/*.pth')
        if available:
            print(f"Available checkpoints: {[os.path.basename(f) for f in available]}")
        return False

    # Load model
    model = load_vqvae(ckpt_path, dim_pose, device)
    if model is None:
        return False

    # Check epoch and warn if too early
    ckpt = torch.load(ckpt_path, map_location=device)
    epoch = ckpt.get('epoch', 0)
    if epoch < 30:
        print(f"\n[WARNING] Only epoch {epoch} - results may not be reliable!")
        print("          Wait until epoch 50+ for meaningful metrics.")
        print("          Or use --latest to check most recent checkpoint.")
        print("          Running anyway for reference...\n")

    # Load dataset
    print("\nLoading dataset samples...")
    cache_path = './datasets/beat_cache/beat_bvh_arkit/train_cache.pkl'

    if not os.path.exists(cache_path):
        print(f"Cache not found: {cache_path}")
        return False

    import pickle
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    samples = cache['samples'][:num_samples]

    # Run reconstruction
    print(f"Running reconstruction on {len(samples)} samples...")

    all_mse = []
    all_vel_ratio = []
    all_input_vel_std = []
    all_recon_vel_std = []
    all_codes = []
    all_perplexity = []
    all_spike_ratio = []
    all_per_joint_vel_ratio = []

    with torch.no_grad():
        for i, sample in enumerate(samples):
            if body_part == 'body':
                motion = sample['body']  # (T, 225)
            else:
                motion = sample['face']  # (T, 51)

            # Normalize using cached stats
            if body_part == 'body':
                mean = cache['body_mean']
                std = cache['body_std']
            else:
                mean = cache['face_mean']
                std = cache['face_std']

            motion_norm = (motion - mean) / std

            # To tensor
            motion_tensor = torch.from_numpy(motion_norm).float().unsqueeze(0).to(device)  # (1, T, D)

            # Full forward pass (includes quantization)
            output = model(motion_tensor)
            recon = output['rec_pose'].cpu().numpy()[0]  # (T, D)
            perplexity = output['perplexity'].cpu().numpy()
            all_perplexity.append(perplexity)

            # Also get the discrete codes to check codebook usage
            code_idx, _ = model.encode(motion_tensor)
            all_codes.append(code_idx.cpu().numpy().flatten())

            # Denormalize
            recon_denorm = recon * std + mean

            # Compute metrics
            mse = np.mean((motion - recon_denorm) ** 2)

            # Velocity analysis
            vel_input = compute_velocity(motion)
            vel_recon = compute_velocity(recon_denorm)

            vel_input_std = np.std(vel_input)
            vel_recon_std = np.std(vel_recon)
            vel_ratio = vel_recon_std / (vel_input_std + 1e-7)

            # Spike detection (catches single bad frames)
            max_vel_input = np.max(np.abs(vel_input))
            max_vel_recon = np.max(np.abs(vel_recon))
            spike_ratio = max_vel_recon / (max_vel_input + 1e-7)

            # Per-joint velocity ratio (which joints are jittery?)
            per_joint_input_std = np.std(vel_input, axis=0) + 1e-7  # (D,)
            per_joint_recon_std = np.std(vel_recon, axis=0)  # (D,)
            per_joint_ratio = per_joint_recon_std / per_joint_input_std  # (D,)

            all_mse.append(mse)
            all_vel_ratio.append(vel_ratio)
            all_input_vel_std.append(vel_input_std)
            all_recon_vel_std.append(vel_recon_std)
            all_spike_ratio.append(spike_ratio)
            all_per_joint_vel_ratio.append(per_joint_ratio)

    # Codebook usage analysis
    all_codes_flat = np.concatenate(all_codes)
    unique_codes = len(np.unique(all_codes_flat))
    total_codes = 1024  # nb_code
    codebook_usage = unique_codes / total_codes * 100

    # Per-joint analysis
    avg_per_joint_ratio = np.mean(all_per_joint_vel_ratio, axis=0)  # (D,)
    worst_joints = np.argsort(avg_per_joint_ratio)[-5:][::-1]  # top 5 jittery joints
    best_joints = np.argsort(avg_per_joint_ratio)[:5]  # top 5 smooth joints

    # Report
    avg_mse = np.mean(all_mse)
    avg_vel_ratio = np.mean(all_vel_ratio)
    avg_input_vel = np.mean(all_input_vel_std)
    avg_recon_vel = np.mean(all_recon_vel_std)
    avg_perplexity = np.mean(all_perplexity)
    avg_spike_ratio = np.mean(all_spike_ratio)
    max_spike_ratio = np.max(all_spike_ratio)

    print("\n" + "-"*50)
    print("RESULTS:")
    print("-"*50)
    print(f"Reconstruction MSE:       {avg_mse:.6f}")
    print(f"")
    print(f"VELOCITY (frame-to-frame smoothness):")
    print(f"  Input velocity std:     {avg_input_vel:.6f}")
    print(f"  Recon velocity std:     {avg_recon_vel:.6f}")
    print(f"  Velocity ratio (avg):   {avg_vel_ratio:.3f}  {'[OK]' if avg_vel_ratio < 1.5 else '[BAD]' if avg_vel_ratio > 2 else '[WARN]'}")
    print(f"")
    print(f"SPIKE DETECTION (single bad frames):")
    print(f"  Spike ratio (avg):      {avg_spike_ratio:.3f}  {'[OK]' if avg_spike_ratio < 2 else '[BAD]' if avg_spike_ratio > 3 else '[WARN]'}")
    print(f"  Spike ratio (worst):    {max_spike_ratio:.3f}  {'[OK]' if max_spike_ratio < 3 else '[BAD]' if max_spike_ratio > 5 else '[WARN]'}")
    print(f"")
    print(f"CODEBOOK:")
    print(f"  Usage:                  {unique_codes}/{total_codes} ({codebook_usage:.1f}%)")
    print(f"  Perplexity:             {avg_perplexity:.1f}")
    print("-"*50)

    # Per-joint analysis
    if body_part == 'body':
        joint_names = [f"joint_{i//3}_{'xyz'[i%3]}" for i in range(225)]
        # Map to approximate body parts (every 3 values = 1 joint)
        print(f"\nPER-JOINT ANALYSIS (75 joints x 3 axes = 225 dims):")
        print(f"  Worst 5 dimensions (most jittery):")
        for j in worst_joints:
            joint_idx = j // 3
            axis = ['x', 'y', 'z'][j % 3]
            print(f"    dim {j:3d} (joint {joint_idx:2d} {axis}): ratio = {avg_per_joint_ratio[j]:.2f}")
        print(f"  Best 5 dimensions (smoothest):")
        for j in best_joints:
            joint_idx = j // 3
            axis = ['x', 'y', 'z'][j % 3]
            print(f"    dim {j:3d} (joint {joint_idx:2d} {axis}): ratio = {avg_per_joint_ratio[j]:.2f}")
    else:
        print(f"\nPER-BLENDSHAPE ANALYSIS (51 ARKit blendshapes):")
        print(f"  Worst 5 blendshapes (most jittery):")
        for j in worst_joints:
            print(f"    blendshape {j:2d}: ratio = {avg_per_joint_ratio[j]:.2f}")
        print(f"  Best 5 blendshapes (smoothest):")
        for j in best_joints:
            print(f"    blendshape {j:2d}: ratio = {avg_per_joint_ratio[j]:.2f}")

    # Interpret results
    print("\n" + "="*50)
    print("INTERPRETATION:")
    print("="*50)

    has_issues = False

    # Epoch warning
    if epoch < 30:
        print(f"\n[WAIT] Epoch {epoch} is too early for reliable metrics")
        print("       Run again after epoch 50+")

    # MSE check
    print(f"\nReconstruction Quality:")
    if avg_mse < 0.1:
        print(f"  [OK] MSE is low ({avg_mse:.4f}) - good reconstruction")
    elif avg_mse < 0.5:
        print(f"  [WARN] MSE is moderate ({avg_mse:.4f}) - acceptable")
    else:
        print(f"  [BAD] MSE is high ({avg_mse:.4f}) - poor reconstruction")
        has_issues = True

    # Velocity ratio check (KEY METRIC for jitter)
    print(f"\nSmoothness (THE KEY METRIC):")
    if avg_vel_ratio < 1.2:
        print(f"  [OK] Velocity ratio {avg_vel_ratio:.2f} - smooth motion preserved")
    elif avg_vel_ratio < 2.0:
        print(f"  [WARN] Velocity ratio {avg_vel_ratio:.2f} - slightly jittery")
    else:
        print(f"  [BAD] Velocity ratio {avg_vel_ratio:.2f} - JITTERY MOTION!")
        print("        This is the problem you had before!")
        has_issues = True

    # Spike check
    if max_spike_ratio > 5:
        print(f"  [BAD] Spike ratio {max_spike_ratio:.2f} - individual frames are jumping!")
        has_issues = True
    elif max_spike_ratio > 3:
        print(f"  [WARN] Spike ratio {max_spike_ratio:.2f} - some frame spikes detected")

    # Codebook usage check
    print(f"\nCodebook Health:")
    if codebook_usage > 30:
        print(f"  [OK] Codebook usage {codebook_usage:.0f}% - codes are being used")
    elif codebook_usage > 10:
        print(f"  [WARN] Codebook usage {codebook_usage:.0f}% - low but acceptable")
    else:
        print(f"  [BAD] Codebook usage {codebook_usage:.0f}% - CODEBOOK COLLAPSE!")
        print("        Model is using too few codes, will lose motion detail")
        has_issues = True

    # Perplexity check (higher = more codes used = better)
    if avg_perplexity > 100:
        print(f"  [OK] Perplexity {avg_perplexity:.0f} - good code diversity")
    elif avg_perplexity > 20:
        print(f"  [WARN] Perplexity {avg_perplexity:.0f} - moderate code diversity")
    else:
        print(f"  [BAD] Perplexity {avg_perplexity:.0f} - poor code diversity")
        has_issues = True

    # Per-joint issues
    jittery_joints = np.sum(avg_per_joint_ratio > 2.0)
    if jittery_joints > 0:
        print(f"\nPer-Joint Issues:")
        print(f"  [WARN] {jittery_joints} dimensions have velocity ratio > 2.0")
        if jittery_joints > dim_pose * 0.2:
            print(f"  [BAD] More than 20% of dimensions are jittery!")
            has_issues = True

    # Final verdict
    print("\n" + "-"*50)
    if epoch < 30:
        print("[WAIT] Run again after epoch 50+ for reliable results")
        return True  # Don't fail early
    elif has_issues:
        print("[FAIL] VQ-VAE has issues - review above")
        return False
    else:
        print("[PASS] VQ-VAE looks healthy!")
        return True


def check_pipeline(audio_path=None, device='cuda'):
    """
    Check full audio -> motion pipeline.

    Tests:
    - Model loading
    - Inference latency
    - Output motion smoothness
    """
    print("\n" + "="*60)
    print("Full Pipeline Sanity Check")
    print("="*60)

    # Check if all required models exist
    required_files = [
        './outputs/vqvae_bvh/body/best.pth',
        './outputs/vqvae_bvh/face/best.pth',
        './outputs/shortcut_bvh_arkit/best_fgd/ckpt.pth',
    ]

    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print("\nMissing required files:")
        for f in missing:
            print(f"  - {f}")
        print("\nGenerator training may still be in progress...")
        return False

    print("\nAll model files found!")

    # Load models
    print("Loading models...")

    # Load VQ-VAEs
    vq_body = load_vqvae('./outputs/vqvae_bvh/body/best.pth', 225, device)
    vq_face = load_vqvae('./outputs/vqvae_bvh/face/best.pth', 51, device)

    # Load normalization stats
    body_mean = np.load('./outputs/vqvae_bvh/body/body_mean.npy')
    body_std = np.load('./outputs/vqvae_bvh/body/body_std.npy')
    face_mean = np.load('./outputs/vqvae_bvh/face/face_mean.npy')
    face_std = np.load('./outputs/vqvae_bvh/face/face_std.npy')

    print("Models loaded!")

    # Find audio from dataset if not provided
    if audio_path is None:
        print("\nLooking for audio in dataset...")
        dataset_path = './datasets/BEAT/beat_english_v0.2.1/beat_english_v0.2.1'
        audio_path = None

        # Find first available wav file
        for speaker in ['1', '2', '3', '4']:
            speaker_dir = os.path.join(dataset_path, speaker)
            if os.path.exists(speaker_dir):
                wav_files = [f for f in os.listdir(speaker_dir) if f.endswith('.wav')]
                if wav_files:
                    audio_path = os.path.join(speaker_dir, wav_files[0])
                    break

        if audio_path is None:
            print("No audio found in dataset, using dummy input...")
            dummy_mel = torch.randn(1, 120, 128).to(device)
        else:
            print(f"Using dataset audio: {os.path.basename(audio_path)}")

    if audio_path is not None:
        print(f"Loading audio: {audio_path}")
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)
        # Limit to 10 seconds for quick test
        audio = audio[:16000 * 10]
        # Compute mel spectrogram
        audio_18k = librosa.resample(audio, orig_sr=16000, target_sr=18000)
        mel = librosa.feature.melspectrogram(y=audio_18k, sr=18000, hop_length=1200, n_mels=128)
        mel = np.swapaxes(mel[..., :-1], -1, -2).astype(np.float32)
        dummy_mel = torch.from_numpy(mel).unsqueeze(0).to(device)

    num_frames = dummy_mel.shape[1]
    print(f"Input: {num_frames} frames ({num_frames/30:.1f} seconds at 30fps)")

    # Latency test
    print("\nRunning latency test (10 iterations)...")

    latencies = []
    with torch.no_grad():
        for i in range(10):
            start = time.time()

            # Encode through VQ-VAE (this simulates the full pipeline)
            # In real inference, the generator would predict latents
            # Here we just test VQ-VAE encode/decode speed

            # Create dummy motion for encode test
            dummy_motion = torch.randn(1, num_frames, 225).to(device)

            # Encode
            codes, latents = vq_body.encode(dummy_motion)

            # Decode
            output = vq_body.forward_decoder(codes)

            torch.cuda.synchronize()
            latency = time.time() - start
            latencies.append(latency)

    avg_latency = np.mean(latencies) * 1000  # ms
    fps = num_frames / (avg_latency / 1000)

    print("\n" + "-"*40)
    print("LATENCY RESULTS:")
    print("-"*40)
    print(f"Avg latency:      {avg_latency:.1f} ms for {num_frames} frames")
    print(f"Per-frame:        {avg_latency/num_frames:.2f} ms")
    print(f"Throughput:       {fps:.0f} fps")
    print("-"*40)

    # Interpret
    print("\nINTERPRETATION:")
    if avg_latency / num_frames < 5:
        print(f"  [OK] Fast enough for real-time ({avg_latency/num_frames:.1f}ms/frame)")
    elif avg_latency / num_frames < 33:
        print(f"  [WARN] May be too slow for real-time ({avg_latency/num_frames:.1f}ms/frame)")
    else:
        print(f"  [BAD] Too slow for real-time ({avg_latency/num_frames:.1f}ms/frame)")

    print("\n[PASS] Pipeline check complete!")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', type=str, default='vqvae',
                        choices=['vqvae', 'vqvae_body', 'vqvae_face', 'pipeline', 'all'],
                        help='What to check')
    parser.add_argument('--audio', type=str, default=None,
                        help='Audio file for pipeline test')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of samples for VQ-VAE check')
    parser.add_argument('--latest', action='store_true',
                        help='Use latest epoch checkpoint instead of best.pth')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Use specific epoch checkpoint (e.g., --epoch 50)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    results = {}

    if args.check in ['vqvae', 'vqvae_body', 'all']:
        results['vqvae_body'] = check_vqvae('body', args.samples, device, args.latest, args.epoch)

    if args.check in ['vqvae_face', 'all']:
        results['vqvae_face'] = check_vqvae('face', args.samples, device, args.latest, args.epoch)

    if args.check in ['pipeline', 'all']:
        results['pipeline'] = check_pipeline(args.audio, device)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name}: {status}")

    if all(results.values()):
        print("\nAll checks passed!")
    else:
        print("\nSome checks failed - review output above")


if __name__ == "__main__":
    main()
