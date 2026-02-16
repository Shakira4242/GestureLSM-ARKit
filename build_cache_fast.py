#!/usr/bin/env python3
"""
FAST parallel cache builder for BEAT BVH + ARKit dataset.
Uses multiprocessing to process multiple files simultaneously.

~15-20x faster than single-threaded on 32-core machines.

Usage:
    python build_cache_fast.py --speakers 1 2 3 4 --workers 16
    python build_cache_fast.py --speakers $(seq 1 30) --workers 32  # All 30 speakers

Output format matches beat_normalized.py dataloader expectations.
"""
import os
import sys
import json
import pickle
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import torch
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloaders.beat_normalized import parse_bvh, extract_joint_rotations, BEAT_JOINTS
from dataloaders.rotation_converter import euler_angles_to_axis_angle


def process_single_file(args_tuple):
    """Process a single BVH/JSON/WAV triplet. Runs in worker process."""
    bvh_path, json_path, wav_path, pose_fps, audio_sr, pose_length, stride = args_tuple

    # Import inside worker to avoid pickling issues
    import librosa

    try:
        # 1. Parse BVH
        joints, joint_channels, frames, frame_time = parse_bvh(bvh_path)
        bvh_fps = 1.0 / frame_time
        rotations = extract_joint_rotations(joints, joint_channels, frames, BEAT_JOINTS)

        # Resample body to target FPS
        if abs(bvh_fps - pose_fps) > 1:
            ratio = bvh_fps / pose_fps
            indices = np.arange(0, len(rotations), ratio).astype(int)
            indices = indices[indices < len(rotations)]
            rotations = rotations[indices]

        # Convert Euler → axis-angle
        euler_radians = np.deg2rad(rotations)
        euler_tensor = torch.from_numpy(euler_radians).float()
        axis_angle_tensor = euler_angles_to_axis_angle(euler_tensor, "XYZ")
        body = axis_angle_tensor.numpy().reshape(len(axis_angle_tensor), -1)

        # 2. Load face JSON
        with open(json_path, 'r') as f:
            face_data = json.load(f)
        face_frames = [frame['weights'] for frame in face_data['frames']]
        face_60fps = np.array(face_frames, dtype=np.float32)

        # Resample face to pose_fps
        face_ratio = 60 / pose_fps
        face_indices = np.arange(0, len(face_60fps), face_ratio).astype(int)
        face_indices = face_indices[face_indices < len(face_60fps)]
        face = face_60fps[face_indices]

        # Align lengths
        min_len = min(len(body), len(face))
        body = body[:min_len]
        face = face[:min_len]

        # 3. Load audio and compute mel
        audio, sr = librosa.load(wav_path, sr=audio_sr)
        audio_18k = librosa.resample(audio, orig_sr=audio_sr, target_sr=18000)
        mel = librosa.feature.melspectrogram(
            y=audio_18k, sr=18000, hop_length=1200, n_mels=128
        )
        mel = np.swapaxes(mel[..., :-1], -1, -2).astype(np.float32)

        # 4. Slice into windows
        samples = []
        for start in range(0, min_len - pose_length, stride):
            end = start + pose_length
            body_slice = body[start:end].astype(np.float32)
            face_slice = face[start:end].astype(np.float32)
            mel_slice = mel[start:end] if end <= len(mel) else np.zeros((pose_length, 128), dtype=np.float32)

            if len(mel_slice) < pose_length:
                mel_slice = np.pad(mel_slice, ((0, pose_length - len(mel_slice)), (0, 0)))

            samples.append({
                'body': body_slice,
                'face': face_slice,
                'mel': mel_slice,
            })

        return samples, None

    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


def build_cache_parallel(
    data_path: str,
    cache_path: str,
    speakers: list,
    pose_fps: int = 30,
    audio_sr: int = 16000,
    pose_length: int = 64,
    stride: int = 10,
    num_workers: int = None,
    split: str = 'train'
):
    """Build cache using parallel processing."""

    if num_workers is None:
        num_workers = min(mp.cpu_count(), 32)

    print(f"=" * 60)
    print(f"Fast Parallel Cache Builder")
    print(f"=" * 60)
    print(f"Workers: {num_workers}")
    print(f"Speakers: {speakers}")
    print(f"Data path: {data_path}")
    print(f"Cache path: {cache_path}")

    # Collect all file triplets
    file_args = []
    for speaker_id in speakers:
        speaker_dir = os.path.join(data_path, str(speaker_id))
        if not os.path.exists(speaker_dir):
            print(f"Warning: Speaker {speaker_id} not found at {speaker_dir}")
            continue

        bvh_files = sorted([f for f in os.listdir(speaker_dir) if f.endswith('.bvh')])
        print(f"Speaker {speaker_id}: {len(bvh_files)} files")

        for bvh_file in bvh_files:
            base_name = bvh_file.replace('.bvh', '')
            bvh_path = os.path.join(speaker_dir, bvh_file)
            json_path = os.path.join(speaker_dir, base_name + '.json')
            wav_path = os.path.join(speaker_dir, base_name + '.wav')

            if os.path.exists(json_path) and os.path.exists(wav_path):
                file_args.append((
                    bvh_path, json_path, wav_path,
                    pose_fps, audio_sr, pose_length, stride
                ))

    if not file_args:
        print("ERROR: No valid file triplets found!")
        return 0

    print(f"\nProcessing {len(file_args)} files with {num_workers} workers...")

    # Process in parallel
    all_samples = []
    errors = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_file, args): args[0] for args in file_args}

        with tqdm(total=len(futures), desc="Building cache") as pbar:
            for future in as_completed(futures):
                samples, error = future.result()
                if error:
                    errors.append((futures[future], error))
                else:
                    all_samples.extend(samples)
                pbar.update(1)

    if errors:
        print(f"\n{len(errors)} files failed:")
        for path, err in errors[:5]:
            print(f"  {os.path.basename(path)}: {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    if not all_samples:
        print("ERROR: No samples generated!")
        return 0

    print(f"\nTotal samples: {len(all_samples)}")

    # Compute normalization stats
    print("Computing normalization stats...")
    all_body = np.stack([s['body'] for s in all_samples])
    all_face = np.stack([s['face'] for s in all_samples])

    body_mean = all_body.mean(axis=(0, 1)).astype(np.float32)
    body_std = all_body.std(axis=(0, 1)).astype(np.float32)
    body_std[body_std < 1e-6] = 1.0

    face_mean = all_face.mean(axis=(0, 1)).astype(np.float32)
    face_std = all_face.std(axis=(0, 1)).astype(np.float32)
    face_std[face_std < 1e-6] = 1.0

    # Save cache (matches beat_normalized.py format exactly)
    os.makedirs(cache_path, exist_ok=True)
    cache_file = os.path.join(cache_path, f'{split}_cache.pkl')

    print(f"Saving cache to {cache_file}...")
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'samples': all_samples,
            'body_mean': body_mean,
            'body_std': body_std,
            'face_mean': face_mean,
            'face_std': face_std,
        }, f)

    # Also save stats as numpy (for VQ-VAE and inference)
    np.save(os.path.join(cache_path, 'body_mean.npy'), body_mean)
    np.save(os.path.join(cache_path, 'body_std.npy'), body_std)
    np.save(os.path.join(cache_path, 'face_mean.npy'), face_mean)
    np.save(os.path.join(cache_path, 'face_std.npy'), face_std)

    print(f"\n" + "=" * 60)
    print(f"SUCCESS!")
    print(f"=" * 60)
    print(f"Cache: {cache_file}")
    print(f"Samples: {len(all_samples)}")
    print(f"Body shape per sample: {all_samples[0]['body'].shape}")
    print(f"Face shape per sample: {all_samples[0]['face'].shape}")
    print(f"Mel shape per sample: {all_samples[0]['mel'].shape}")
    print(f"\nNormalization stats saved:")
    print(f"  {cache_path}/body_mean.npy")
    print(f"  {cache_path}/body_std.npy")
    print(f"  {cache_path}/face_mean.npy")
    print(f"  {cache_path}/face_std.npy")

    return len(all_samples)


def main():
    parser = argparse.ArgumentParser(description='Fast parallel cache builder for BEAT dataset')
    parser.add_argument('--data_path', type=str,
                        default='./datasets/BEAT/beat_english_v0.2.1/beat_english_v0.2.1/')
    parser.add_argument('--cache_path', type=str,
                        default='./datasets/beat_cache/beat_bvh_arkit/')
    parser.add_argument('--speakers', type=int, nargs='+', default=[1, 2, 3, 4],
                        help='Speaker IDs to process (e.g., --speakers 1 2 3 4 or --speakers $(seq 1 30))')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count, max 32)')
    parser.add_argument('--pose_length', type=int, default=64)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])

    args = parser.parse_args()

    build_cache_parallel(
        data_path=args.data_path,
        cache_path=args.cache_path,
        speakers=args.speakers,
        num_workers=args.workers,
        pose_length=args.pose_length,
        stride=args.stride,
        split=args.split,
    )


if __name__ == '__main__':
    main()
