#!/usr/bin/env python3
"""
GestureLSM BVH Inference - generates BVH body motion + ARKit face from audio.
Adapted from run_inference.py for BVH format (225D body + 51D face).

Outputs:
  - .bvh file: Body motion in standard BVH format (viewable in Blender)
  - .json file: ARKit 51 blendshapes (for face animation)
  - .npz file: Raw numpy arrays (for further processing)
"""
import os
import sys
import json
import warnings
import torch
import numpy as np
import librosa

if not sys.warnoptions:
    warnings.simplefilter("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloaders.build_vocab import Vocab
sys.modules['__main__'].Vocab = Vocab

from utils import other_tools
from models.vq.model import RVQVAE
from dataloaders.pymo.parsers import BVHParser
from dataloaders.pymo.writers import BVHWriter
from dataloaders import rotation_converter as rc

device = "cuda" if torch.cuda.is_available() else "cpu"

# ARKit 51 blendshape names (standard order from BEAT dataset)
ARKIT_BLENDSHAPES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft",
    "noseSneerRight"
]


def process_audio_to_mel(audio_path, audio_sr=16000, n_mels=128, pose_fps=30):
    """Process audio file into mel spectrogram features matching training format."""
    audio, sr = librosa.load(audio_path, sr=audio_sr)

    # Compute mel spectrogram (same as training)
    hop_length = audio_sr // pose_fps  # 16000/30 = 533
    mel = librosa.feature.melspectrogram(
        y=audio, sr=audio_sr, n_mels=n_mels, hop_length=hop_length
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to roughly [-1, 1] range
    mel_db = (mel_db + 40) / 40  # Shift and scale

    # Transpose to [T, n_mels]
    mel_features = mel_db.T

    duration = len(audio) / audio_sr
    return mel_features, duration, len(audio)


def load_bvh_vqvae_models(args):
    """Load VQ-VAE models for BVH body and face."""
    vq_args = type('Args', (), {
        'mu': 0.99, 'nb_code': 1024, 'code_dim': 128,
        'down_t': 2, 'stride_t': 2, 'width': 512, 'depth': 3,
        'dilation_growth_rate': 3, 'vq_act': 'relu', 'vq_norm': None,
        'num_quantizers': 6, 'shared_codebook': False,
        'quantize_dropout_prob': 0.2, 'quantize_dropout_cutoff_index': 0
    })()

    models = {}

    # Body VQ-VAE (225D axis-angle)
    body_model = RVQVAE(vq_args, 225, vq_args.nb_code, vq_args.code_dim, vq_args.code_dim,
                        vq_args.down_t, vq_args.stride_t, vq_args.width, vq_args.depth,
                        vq_args.dilation_growth_rate, vq_args.vq_act, vq_args.vq_norm)
    ckpt = torch.load(args.vqvae_body_path, map_location=device, weights_only=False)
    body_model.load_state_dict(ckpt['net'])
    body_model.eval().to(device)
    models['body'] = body_model

    # Face VQ-VAE (51D ARKit)
    face_model = RVQVAE(vq_args, 51, vq_args.nb_code, vq_args.code_dim, vq_args.code_dim,
                        vq_args.down_t, vq_args.stride_t, vq_args.width, vq_args.depth,
                        vq_args.dilation_growth_rate, vq_args.vq_act, vq_args.vq_norm)
    ckpt = torch.load(args.vqvae_face_path, map_location=device, weights_only=False)
    face_model.load_state_dict(ckpt['net'])
    face_model.eval().to(device)
    models['face'] = face_model

    return models


def load_bvh_normalization(args):
    """Load normalization stats for BVH body and face."""
    norm = {
        'body_mean': torch.from_numpy(np.load(args.mean_pose_path)).float().to(device),
        'body_std': torch.from_numpy(np.load(args.std_pose_path)).float().to(device),
        'face_mean': torch.from_numpy(np.load(args.mean_face_path)).float().to(device),
        'face_std': torch.from_numpy(np.load(args.std_face_path)).float().to(device),
    }
    return norm


def prepare_seed_from_data(data_path, vq_models, norm, args):
    """Prepare seed latents from ground truth BVH+ARKit data."""
    # Load cached data sample
    import pickle
    cache_path = args.cache_path + '/train_cache.pkl'

    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    # Get first sample for seed - handle different cache structures
    if isinstance(cache, dict):
        # Dict structure - get samples list
        if 'samples' in cache:
            samples = cache['samples']
            sample = samples[0]
        else:
            # Maybe it's a dict with numeric string keys or other structure
            keys = list(cache.keys())
            if 'pose' in keys:
                # Single sample dict
                sample = cache
            else:
                # Try first key
                sample = cache[keys[0]]
    elif isinstance(cache, list):
        sample = cache[0]
    else:
        raise ValueError(f"Unknown cache structure: {type(cache)}")

    # Get body and face data (keys are 'body' and 'face' in BVH cache)
    body_key = 'body' if 'body' in sample else 'pose'
    face_key = 'face' if 'face' in sample else 'facial'
    body = torch.from_numpy(sample[body_key][:args.pose_length]).float().to(device)
    face = torch.from_numpy(sample[face_key][:args.pose_length]).float().to(device)

    # Normalize
    body_norm = (body - norm['body_mean']) / norm['body_std']
    face_norm = (face - norm['face_mean']) / norm['face_std']

    # Add batch dim
    body_norm = body_norm.unsqueeze(0)
    face_norm = face_norm.unsqueeze(0)

    # Encode to latent
    with torch.no_grad():
        lat_body = vq_models['body'].map2latent(body_norm)
        lat_face = vq_models['face'].map2latent(face_norm)

    # Concat and scale
    latent = torch.cat([lat_body, lat_face], dim=2) / args.vqvae_latent_scale
    return latent


def run_bvh_inference(model, vq_models, mel_features, seed_latent, norm, args):
    """Run shortcut inference for BVH motion generation."""
    bs = 1
    squeeze = args.vqvae_squeeze_scale
    pre_frames = args.pre_frames
    pose_len = args.pose_length

    # Prepare audio input [B, T, 128]
    mel_tensor = torch.from_numpy(mel_features).float().unsqueeze(0).to(device)
    n_frames = mel_tensor.shape[1]

    # Align to pose_length chunks
    n_frames = (n_frames // 8) * 8
    if n_frames < pose_len:
        n_frames = pose_len

    print(f"Generating {n_frames} frames...")

    pre_scaled = pre_frames * squeeze
    roundt = max(1, (n_frames - pre_scaled) // (pose_len - pre_scaled))
    round_l = pose_len - pre_scaled

    rec_body, rec_face = [], []
    in_seed = seed_latent
    last_sample = None

    for i in range(roundt):
        # Audio slice - mel is at pose_fps (30), need to slice properly
        a_start = i * round_l
        a_end = (i + 1) * round_l + pre_frames * squeeze
        audio_tmp = mel_tensor[:, a_start:a_end, :]

        # Pad if needed
        if audio_tmp.shape[1] < pose_len:
            pad_size = pose_len - audio_tmp.shape[1]
            audio_tmp = torch.nn.functional.pad(audio_tmp, (0, 0, 0, pad_size))

        # Seed
        if i == 0:
            seed_tmp = in_seed[:, :pre_frames, :]
        else:
            seed_tmp = last_sample[:, -pre_frames:, :]

        cond = {'y': {
            'audio_onset': audio_tmp,  # Model expects 'audio_onset' key
            'word': None,  # No text in BVH mode
            'id': torch.zeros(bs, pre_frames).long().to(device),
            'seed': seed_tmp,
            'style_feature': torch.zeros(bs, 512).to(device),
        }}

        with torch.no_grad():
            output = model(cond)
            sample = output['latents'].squeeze(2).permute(0, 2, 1)

        last_sample = sample.clone()

        # Split body and face latents
        lat_body = sample[..., :128]
        lat_face = sample[..., 128:256]

        if i == 0:
            rec_body.append(lat_body)
            rec_face.append(lat_face)
        else:
            rec_body.append(lat_body[:, pre_frames:])
            rec_face.append(lat_face[:, pre_frames:])

    # Concat and scale
    rec_body = torch.cat(rec_body, dim=1) * args.vqvae_latent_scale
    rec_face = torch.cat(rec_face, dim=1) * args.vqvae_latent_scale

    # Decode
    with torch.no_grad():
        out_body = vq_models['body'].latent2origin(rec_body)[0]
        out_face = vq_models['face'].latent2origin(rec_face)[0]

    # Denormalize
    out_body = out_body * norm['body_std'] + norm['body_mean']
    out_face = out_face * norm['face_std'] + norm['face_mean']

    return out_body, out_face


def save_bvh(body_axis_angle, template_bvh_path, output_path, fps=30):
    """
    Save body motion as BVH file using a template for skeleton structure.

    Args:
        body_axis_angle: [T, 225] axis-angle rotations (75 joints × 3)
        template_bvh_path: Path to a BVH file to use as skeleton template
        output_path: Output BVH file path
        fps: Frame rate
    """
    # Parse template to get skeleton structure
    parser = BVHParser()
    mocap_data = parser.parse(template_bvh_path, start=0, stop=10)  # Just need skeleton

    # Convert axis-angle to Euler angles (BVH uses Euler)
    n_frames, n_dims = body_axis_angle.shape
    n_joints = n_dims // 3

    # Reshape to [T, 75, 3]
    axis_angle = torch.from_numpy(body_axis_angle).float().reshape(n_frames, n_joints, 3)

    # Convert to Euler (XYZ convention for BVH)
    euler_angles = rc.axis_angle_to_euler_angles(axis_angle)  # [T, 75, 3]
    euler_degrees = torch.rad2deg(euler_angles).numpy()  # Convert to degrees

    # Flatten back to [T, 225]
    euler_flat = euler_degrees.reshape(n_frames, -1)

    # Update motion data in mocap_data
    # The BVH format has channels in a specific order matching the skeleton
    import pandas as pd

    # Create new DataFrame with same columns
    columns = mocap_data.values.columns
    time_index = pd.to_timedelta(np.arange(n_frames) / fps, unit='s')

    # Map our joint rotations to BVH channels
    # Note: BVH typically has root position + rotations
    new_data = np.zeros((n_frames, len(columns)))

    # Fill in rotation data (skip first 3 channels which are root position)
    rot_start = 3  # After Xposition, Yposition, Zposition
    for i in range(min(n_joints, (len(columns) - 3) // 3)):
        for j in range(3):
            col_idx = rot_start + i * 3 + j
            if col_idx < len(columns):
                new_data[:, col_idx] = euler_flat[:, i * 3 + j]

    mocap_data.values = pd.DataFrame(data=new_data, index=time_index, columns=columns)
    mocap_data.framerate = 1.0 / fps

    # Write BVH
    writer = BVHWriter()
    with open(output_path, 'w') as f:
        writer.write(mocap_data, f)

    return output_path


def save_arkit_json(face_blendshapes, output_path, fps=30):
    """
    Save ARKit blendshapes as JSON with proper blendshape names.

    Args:
        face_blendshapes: [T, 51] blendshape weights
        output_path: Output JSON file path
        fps: Frame rate
    """
    n_frames = face_blendshapes.shape[0]

    # Build frame-by-frame blendshape data
    frames = []
    for t in range(n_frames):
        frame_data = {
            "time": t / fps,
            "blendshapes": {}
        }
        for i, name in enumerate(ARKIT_BLENDSHAPES):
            frame_data["blendshapes"][name] = float(face_blendshapes[t, i])
        frames.append(frame_data)

    output_data = {
        "fps": fps,
        "n_frames": n_frames,
        "duration": n_frames / fps,
        "blendshape_names": ARKIT_BLENDSHAPES,
        "frames": frames
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    return output_path


def find_template_bvh(data_path):
    """Find a template BVH file from the dataset."""
    import glob

    # Look for any BVH file in the dataset
    patterns = [
        os.path.join(data_path, "**/*.bvh"),
        os.path.join(data_path, "*/*.bvh"),
        os.path.join(data_path, "*.bvh"),
    ]

    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            return files[0]

    return None


def compute_smoothness_metrics(motion, name, expected_range=None):
    """
    Compute smoothness metrics for motion data.
    Returns dict with pass/fail status for each metric.
    """
    results = {'name': name, 'passed': True, 'metrics': {}}

    # Velocity (1st derivative)
    vel = np.diff(motion, axis=0)
    # Acceleration (2nd derivative)
    acc = np.diff(vel, axis=0)
    # Jerk (3rd derivative) - high jerk = jittery
    jerk = np.diff(acc, axis=0)

    # Temporal autocorrelation (how similar consecutive frames are)
    # High autocorr = smooth continuous motion
    autocorr = np.corrcoef(motion[:-1].flatten(), motion[1:].flatten())[0, 1]

    # Max frame-to-frame jump
    max_jump = np.abs(vel).max()

    # Store metrics
    results['metrics']['value_min'] = float(motion.min())
    results['metrics']['value_max'] = float(motion.max())
    results['metrics']['velocity_std'] = float(vel.std())
    results['metrics']['acceleration_std'] = float(acc.std())
    results['metrics']['jerk_std'] = float(jerk.std())
    results['metrics']['temporal_autocorr'] = float(autocorr)
    results['metrics']['max_jump'] = float(max_jump)

    # Check pass/fail criteria
    # Temporal autocorrelation > 0.9 means smooth continuous motion
    if autocorr < 0.9:
        results['passed'] = False
        results['fail_reason'] = f"Low temporal autocorr ({autocorr:.4f} < 0.9) - motion may be discontinuous"

    # Jerk std < 0.5 means not too jittery (threshold depends on data scale)
    if jerk.std() > 0.5:
        results['passed'] = False
        results['fail_reason'] = f"High jerk ({jerk.std():.4f} > 0.5) - motion may be jittery"

    # Check value range if provided
    if expected_range:
        if motion.min() < expected_range[0] or motion.max() > expected_range[1]:
            results['passed'] = False
            results['fail_reason'] = f"Values outside expected range {expected_range}"

    return results


def print_smoothness_report(body_np, face_np):
    """Print smoothness metrics report for body and face motion."""
    print(f"\n{'=' * 60}")
    print("SMOOTHNESS METRICS (Sanity Check)")
    print(f"{'=' * 60}")

    # Body metrics (axis-angle, roughly ±π range)
    body_results = compute_smoothness_metrics(body_np, "BODY", expected_range=(-4, 4))
    print(f"\n{body_results['name']} (225D axis-angle):")
    print(f"  Value range: [{body_results['metrics']['value_min']:.3f}, {body_results['metrics']['value_max']:.3f}]")
    print(f"  Velocity std: {body_results['metrics']['velocity_std']:.6f}")
    print(f"  Acceleration std: {body_results['metrics']['acceleration_std']:.6f}")
    print(f"  Jerk std: {body_results['metrics']['jerk_std']:.6f}")
    print(f"  Temporal autocorr: {body_results['metrics']['temporal_autocorr']:.4f}")
    print(f"  Max frame-to-frame jump: {body_results['metrics']['max_jump']:.4f}")

    if body_results['passed']:
        print(f"  ✓ PASSED - Motion is smooth and continuous")
    else:
        print(f"  ✗ FAILED - {body_results.get('fail_reason', 'Unknown')}")

    # Face metrics (blendshapes, 0-1 range)
    face_results = compute_smoothness_metrics(face_np, "FACE", expected_range=(-0.5, 1.5))
    print(f"\n{face_results['name']} (51D ARKit blendshapes):")
    print(f"  Value range: [{face_results['metrics']['value_min']:.3f}, {face_results['metrics']['value_max']:.3f}]")
    print(f"  Velocity std: {face_results['metrics']['velocity_std']:.6f}")
    print(f"  Acceleration std: {face_results['metrics']['acceleration_std']:.6f}")
    print(f"  Jerk std: {face_results['metrics']['jerk_std']:.6f}")
    print(f"  Temporal autocorr: {face_results['metrics']['temporal_autocorr']:.4f}")
    print(f"  Max frame-to-frame jump: {face_results['metrics']['max_jump']:.4f}")

    if face_results['passed']:
        print(f"  ✓ PASSED - Motion is smooth and continuous")
    else:
        print(f"  ✗ FAILED - {face_results.get('fail_reason', 'Unknown')}")

    # Overall verdict
    print(f"\n{'-' * 60}")
    if body_results['passed'] and face_results['passed']:
        print("OVERALL: ✓ PASSED - Output is smooth and ready for Unity")
    else:
        print("OVERALL: ✗ FAILED - Output may have issues")
        print("  (This is expected early in training - keep training!)")
    print(f"{'=' * 60}")


def find_test_audio(data_path):
    """Find a test audio file from the dataset."""
    import glob

    patterns = [
        os.path.join(data_path, "**/*.wav"),
        os.path.join(data_path, "*/*.wav"),
    ]

    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            return files[0]

    return None


def main():
    import argparse

    # Parse our CLI args BEFORE importing config (which uses configargparse)
    parser = argparse.ArgumentParser(description='GestureLSM BVH Inference', add_help=False)
    parser.add_argument('--audio', type=str, default=None, help='Path to input audio file')
    parser.add_argument('--output', type=str, default='output_bvh_motion.npz', help='Output path')
    parser.add_argument('--config', type=str, default='configs/shortcut_bvh_arkit.yaml')
    parser.add_argument('--checkpoint', type=str, default=None, help='Generator checkpoint')

    # Parse only known args
    cli_args, _ = parser.parse_known_args()

    print("=" * 60)
    print("GestureLSM BVH Inference")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Config: {cli_args.config}")
    print(f"Checkpoint: {cli_args.checkpoint}")
    print(f"Output: {cli_args.output}")

    # Load config directly from YAML (bypass configargparse conflicts)
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(cli_args.config)

    # Create args namespace from config
    class Args:
        pass
    args = Args()

    # Copy all config values to args
    for key, value in OmegaConf.to_container(cfg, resolve=True).items():
        setattr(args, key, value)

    # Set defaults for paths
    args.data_path = getattr(args, 'data_path', './datasets/BEAT/beat_english_v0.2.1/beat_english_v0.2.1/')
    args.cache_path = getattr(args, 'cache_path', './datasets/beat_cache/beat_bvh_arkit/')
    args.out_path = getattr(args, 'out_path', './outputs/shortcut_bvh_arkit/')
    args.pose_length = getattr(args, 'pose_length', 128)
    args.pre_frames = getattr(args, 'pre_frames', 4)
    args.pose_fps = getattr(args, 'pose_fps', 30)
    args.audio_sr = getattr(args, 'audio_sr', 16000)
    args.audio_f = getattr(args, 'audio_f', 128)
    args.vqvae_latent_scale = getattr(args, 'vqvae_latent_scale', 5)
    args.vqvae_squeeze_scale = getattr(args, 'vqvae_squeeze_scale', 4)
    args.g_name = getattr(args, 'g_name', 'GestureLSM') if not hasattr(args, 'model') else cfg.model.g_name

    # Find audio if not provided
    audio_path = cli_args.audio
    if audio_path is None:
        audio_path = find_test_audio(args.data_path)
        if audio_path is None:
            print("ERROR: No audio file provided and couldn't find one in dataset")
            print(f"       Searched in: {args.data_path}")
            sys.exit(1)
        print(f"Using test audio from dataset: {audio_path}")
    else:
        print(f"Audio: {audio_path}")

    # Override checkpoint if provided
    if cli_args.checkpoint:
        args.test_ckpt = cli_args.checkpoint
    else:
        args.test_ckpt = args.out_path + 'best.pth'

    # Load normalization
    print("\n1. Loading normalization stats...")
    norm = load_bvh_normalization(args)
    print(f"   Body: mean shape {norm['body_mean'].shape}, std shape {norm['body_std'].shape}")
    print(f"   Face: mean shape {norm['face_mean'].shape}, std shape {norm['face_std'].shape}")

    # Load VQ-VAEs
    print("\n2. Loading BVH VQ-VAE models...")
    vq_models = load_bvh_vqvae_models(args)
    print(f"   Body VQ-VAE: {args.vqvae_body_path}")
    print(f"   Face VQ-VAE: {args.vqvae_face_path}")

    # Load generator
    print("\n3. Loading shortcut generator...")
    model_module = __import__(f'models.{cfg.model.model_name}', fromlist=['something'])
    model = getattr(model_module, cfg.model.g_name)(cfg)
    model = torch.nn.DataParallel(model, [0]).to(device)

    if os.path.exists(args.test_ckpt):
        # Load checkpoint directly (uses model_state_dict key)
        ckpt = torch.load(args.test_ckpt, map_location=device, weights_only=False)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        elif 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state'])
        else:
            # Try loading directly
            model.load_state_dict(ckpt)
        print(f"   Loaded: {args.test_ckpt}")
        if 'epoch' in ckpt:
            print(f"   Epoch: {ckpt['epoch']}")
    else:
        print(f"   WARNING: Checkpoint not found at {args.test_ckpt}")
        print(f"   Using random weights (for testing pipeline only)")
    model.eval()

    # Process audio
    print("\n4. Processing audio...")
    mel_features, duration, n_samples = process_audio_to_mel(
        audio_path,
        audio_sr=args.audio_sr,
        n_mels=args.audio_f,
        pose_fps=args.pose_fps
    )
    print(f"   Duration: {duration:.2f}s")
    print(f"   Mel features: {mel_features.shape}")

    # Prepare seed
    print("\n5. Preparing seed latents...")
    seed_latent = prepare_seed_from_data(None, vq_models, norm, args)
    print(f"   Seed shape: {seed_latent.shape}")

    # Run inference
    print("\n6. Running shortcut inference...")
    out_body, out_face = run_bvh_inference(
        model, vq_models, mel_features, seed_latent, norm, args
    )
    print(f"   Body output: {out_body.shape}")
    print(f"   Face output: {out_face.shape}")

    # Save outputs
    print("\n7. Saving outputs...")
    n_out = out_body.shape[1]
    body_np = out_body.squeeze(0).cpu().numpy()
    face_np = out_face.squeeze(0).cpu().numpy()

    # Determine output paths
    base_path = cli_args.output.replace('.npz', '')
    npz_path = base_path + '.npz'
    bvh_path = base_path + '.bvh'
    json_path = base_path + '_arkit.json'

    # Save NPZ (raw data + audio path for playback)
    np.savez(npz_path,
        body=body_np,  # [T, 225] axis-angle
        face=face_np,  # [T, 51] ARKit blendshapes
        fps=args.pose_fps,
        format='bvh_arkit',
        body_dims=225,
        face_dims=51,
        audio_path=audio_path,  # Source audio for sync playback
    )
    print(f"   NPZ saved: {npz_path}")

    # Save ARKit JSON (easy to visualize)
    save_arkit_json(face_np, json_path, fps=args.pose_fps)
    print(f"   ARKit JSON saved: {json_path}")

    # Try to save BVH (needs template)
    template_bvh = find_template_bvh(args.data_path)
    if template_bvh:
        try:
            save_bvh(body_np, template_bvh, bvh_path, fps=args.pose_fps)
            print(f"   BVH saved: {bvh_path}")
        except Exception as e:
            print(f"   BVH export failed: {e}")
            print(f"   (You can still use the NPZ file)")
    else:
        print(f"   BVH export skipped (no template found in {args.data_path})")

    # Print preview of first few frames
    print(f"\n{'=' * 60}")
    print("PREVIEW - First frame values:")
    print(f"  Body (first 12 values): {body_np[0, :12].round(3)}")
    print(f"  Face (first 6 blendshapes):")
    for i in range(min(6, len(ARKIT_BLENDSHAPES))):
        print(f"    {ARKIT_BLENDSHAPES[i]}: {face_np[0, i]:.4f}")

    # Run smoothness sanity check
    print_smoothness_report(body_np, face_np)

    print(f"\n{'=' * 60}")
    print(f"SUCCESS!")
    print(f"  Frames: {n_out} ({n_out/args.pose_fps:.2f}s at {args.pose_fps}fps)")
    print(f"  Body: {body_np.shape} (75 joints × 3 axis-angle)")
    print(f"  Face: {face_np.shape} (51 ARKit blendshapes)")
    print(f"\nOutputs:")
    print(f"  {npz_path} - Raw numpy arrays")
    print(f"  {json_path} - ARKit blendshapes (open in any JSON viewer)")
    if template_bvh:
        print(f"  {bvh_path} - Body motion (open in Blender)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
