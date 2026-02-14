#!/usr/bin/env python3
"""
GestureLSM MeanFlow Inference - generates motion NPZ from audio.
Based on demo.py but without Gradio/Whisper/MFA dependencies.
"""
import os
import sys
import warnings
import torch
import numpy as np
import librosa
from numpy.lib import stride_tricks

if not sys.warnoptions:
    warnings.simplefilter("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloaders.build_vocab import Vocab
sys.modules['__main__'].Vocab = Vocab

from utils import config, other_tools
from utils import rotation_conversions as rc
from utils.joints import upper_body_mask, hands_body_mask, lower_body_mask
from dataloaders.data_tools import joints_list
from models.vq.model import RVQVAE

device = "cuda" if torch.cuda.is_available() else "cpu"


def process_audio_to_onset(audio_path, audio_sr=16000):
    """Process audio file into onset+amplitude features."""
    audio, sr = librosa.load(audio_path)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=audio_sr)

    # Compute amplitude envelope
    frame_length = 1024
    shape = (audio.shape[-1] - frame_length + 1, frame_length)
    strides = (audio.strides[-1], audio.strides[-1])
    rolling_view = stride_tricks.as_strided(audio, shape=shape, strides=strides)
    amplitude = np.max(np.abs(rolling_view), axis=1)
    amplitude = np.pad(amplitude, (0, frame_length-1), mode='constant', constant_values=amplitude[-1])

    # Compute onset
    onset_frames = librosa.onset.onset_detect(y=audio, sr=audio_sr, units='frames')
    onset = np.zeros(len(audio), dtype=float)
    onset[onset_frames] = 1.0

    # Combine: [N, 2]
    features = np.concatenate([amplitude.reshape(-1, 1), onset.reshape(-1, 1)], axis=1)
    return features, len(audio) / audio_sr


def load_vqvae_models(args):
    """Load VQ-VAE models for body parts."""
    vq_args = type('Args', (), {
        'mu': 0.99, 'nb_code': 1024, 'code_dim': 128,
        'down_t': 2, 'stride_t': 2, 'width': 512, 'depth': 3,
        'dilation_growth_rate': 3, 'vq_act': 'relu', 'vq_norm': None,
        'num_quantizers': 6, 'shared_codebook': False,
        'quantize_dropout_prob': 0.2, 'quantize_dropout_cutoff_index': 0
    })()

    configs = {
        'upper': (78, args.vqvae_upper_path),
        'hands': (180, args.vqvae_hands_path),
        'lower': (57, args.vqvae_lower_path),
    }

    models = {}
    for name, (dim, path) in configs.items():
        model = RVQVAE(vq_args, dim, vq_args.nb_code, vq_args.code_dim, vq_args.code_dim,
                       vq_args.down_t, vq_args.stride_t, vq_args.width, vq_args.depth,
                       vq_args.dilation_growth_rate, vq_args.vq_act, vq_args.vq_norm)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['net'])
        model.eval().to(device)
        models[name] = model

    return models


def prepare_seed_latent(npz_path, vq_models, norm, args):
    """Prepare seed latents from ground truth poses."""
    data = np.load(npz_path, allow_pickle=True)
    poses = torch.from_numpy(data['poses']).float().to(device)
    trans = torch.from_numpy(data['trans']).float().to(device)

    trans_v = torch.zeros_like(trans)
    trans_v[1:] = trans[1:] - trans[:-1]

    n = min(args.pose_length, poses.shape[0])
    poses = poses[:n].unsqueeze(0)
    trans_v = trans_v[:n].unsqueeze(0)
    bs = 1

    # Build joint masks
    ori_joints = joints_list['beat_smplx_joints']
    masks = {}
    for name, tar_list in [('upper', joints_list["beat_smplx_upper"]),
                           ('hands', joints_list["beat_smplx_hands"]),
                           ('lower', joints_list["beat_smplx_lower"])]:
        mask = np.zeros(len(list(ori_joints.keys()))*3)
        for jn in tar_list:
            mask[ori_joints[jn][1] - ori_joints[jn][0]:ori_joints[jn][1]] = 1
        masks[name] = mask

    # Convert to 6D rotation
    tar_hands = poses[:, :, 25*3:55*3]
    tar_hands = rc.axis_angle_to_matrix(tar_hands.reshape(bs, n, 30, 3))
    tar_hands = rc.matrix_to_rotation_6d(tar_hands).reshape(bs, n, 180)

    tar_upper = poses[:, :, masks['upper'].astype(bool)]
    tar_upper = rc.axis_angle_to_matrix(tar_upper.reshape(bs, n, 13, 3))
    tar_upper = rc.matrix_to_rotation_6d(tar_upper).reshape(bs, n, 78)

    tar_leg = poses[:, :, masks['lower'].astype(bool)]
    tar_leg = rc.axis_angle_to_matrix(tar_leg.reshape(bs, n, 9, 3))
    tar_leg = rc.matrix_to_rotation_6d(tar_leg).reshape(bs, n, 54)

    # Normalize
    tar_upper = (tar_upper - norm['mean_upper']) / norm['std_upper']
    tar_hands = (tar_hands - norm['mean_hands']) / norm['std_hands']
    tar_lower = (tar_leg - norm['mean_lower']) / norm['std_lower']

    # Add trans_v
    trans_v_norm = (trans_v - norm['trans_mean']) / norm['trans_std']
    tar_lower = torch.cat([tar_lower, trans_v_norm], dim=-1)

    # Encode to latent
    with torch.no_grad():
        lat_upper = vq_models['upper'].map2latent(tar_upper)
        lat_hands = vq_models['hands'].map2latent(tar_hands)
        lat_lower = vq_models['lower'].map2latent(tar_lower)

    latent = torch.cat([lat_upper, lat_hands, lat_lower], dim=2) / 5.0
    return latent, masks


def run_inference(model, vq_models, audio_onset, seed_latent, norm, args):
    """Run MeanFlow inference."""
    bs = 1
    squeeze = 4
    pre_frames = args.pre_frames
    pose_len = args.pose_length

    audio_onset = torch.from_numpy(audio_onset).float().unsqueeze(0).to(device)
    audio_samples = audio_onset.shape[1]
    n_frames = audio_samples * 30 // 16000
    n_frames = (n_frames // 8) * 8
    if n_frames < pose_len:
        n_frames = pose_len

    print(f"Generating {n_frames} frames...")

    pre_scaled = pre_frames * squeeze
    roundt = max(1, (n_frames - pre_scaled) // (pose_len - pre_scaled))
    round_l = pose_len - pre_scaled

    rec_upper, rec_hands, rec_lower = [], [], []
    in_seed = seed_latent
    # Word tokens - use PAD token (0) since we don't have TextGrid alignment
    in_word = torch.zeros(bs, n_frames).long().to(device)
    last_sample = None

    for i in range(roundt):
        # Audio slice
        a_start = i * (16000 // 30 * round_l)
        a_end = (i + 1) * (16000 // 30 * round_l) + 16000 // 30 * pre_frames * squeeze
        audio_tmp = audio_onset[:, a_start:a_end, :]

        # Word slice (1D token indices)
        w_start = i * round_l
        w_end = (i + 1) * round_l + pre_frames * squeeze
        word_tmp = in_word[:, w_start:w_end]

        # Seed
        if i == 0:
            seed_tmp = in_seed[:, :pre_frames, :]
        else:
            seed_tmp = last_sample[:, -pre_frames:, :]

        cond = {'y': {
            'audio_onset': audio_tmp,
            'word': word_tmp,
            'id': torch.zeros(bs, pre_frames).long().to(device),
            'seed': seed_tmp,
            'style_feature': torch.zeros(bs, 512).to(device),
        }}

        with torch.no_grad():
            output = model(cond)
            sample = output['latents'].squeeze(2).permute(0, 2, 1)

        last_sample = sample.clone()

        # Split
        lat_upper = sample[..., :128]
        lat_hands = sample[..., 128:256]
        lat_lower = sample[..., 256:384]

        if i == 0:
            rec_upper.append(lat_upper)
            rec_hands.append(lat_hands)
            rec_lower.append(lat_lower)
        else:
            rec_upper.append(lat_upper[:, pre_frames:])
            rec_hands.append(lat_hands[:, pre_frames:])
            rec_lower.append(lat_lower[:, pre_frames:])

    # Concat and scale
    rec_upper = torch.cat(rec_upper, dim=1) * 5.0
    rec_hands = torch.cat(rec_hands, dim=1) * 5.0
    rec_lower = torch.cat(rec_lower, dim=1) * 5.0

    # Decode
    with torch.no_grad():
        out_upper = vq_models['upper'].latent2origin(rec_upper)[0]
        out_hands = vq_models['hands'].latent2origin(rec_hands)[0]
        out_lower = vq_models['lower'].latent2origin(rec_lower)[0]

    # Extract trans
    trans_v = out_lower[..., -3:]
    trans_v = trans_v * norm['trans_std'] + norm['trans_mean']
    trans = torch.cumsum(trans_v, dim=1)
    trans[..., 1] = trans_v[..., 1]
    out_lower = out_lower[..., :-3]

    # Denormalize
    out_upper = out_upper * norm['std_upper'] + norm['mean_upper']
    out_hands = out_hands * norm['std_hands'] + norm['mean_hands']
    out_lower = out_lower * norm['std_lower'] + norm['mean_lower']

    return out_upper, out_hands, out_lower, trans


def convert_to_poses(out_upper, out_hands, out_lower, masks):
    """Convert 6D rotations to axis-angle poses."""
    bs, n = out_upper.shape[0], out_upper.shape[1]

    # Upper: 13 joints
    pose_upper = out_upper.reshape(bs, n, 13, 6)
    pose_upper = rc.rotation_6d_to_matrix(pose_upper)
    pose_upper = rc.matrix_to_axis_angle(pose_upper).reshape(bs*n, 39)

    # Hands: 30 joints
    pose_hands = out_hands.reshape(bs, n, 30, 6)
    pose_hands = rc.rotation_6d_to_matrix(pose_hands)
    pose_hands = rc.matrix_to_axis_angle(pose_hands).reshape(bs*n, 90)

    # Lower: 9 joints
    pose_lower = out_lower.reshape(bs, n, 9, 6)
    pose_lower = rc.rotation_6d_to_matrix(pose_lower)
    pose_lower = rc.matrix_to_axis_angle(pose_lower).reshape(bs*n, 27)

    # Reconstruct full 55-joint pose
    full_pose = torch.zeros(bs*n, 165).to(out_upper.device)

    # Map joints back
    upper_idx = torch.where(torch.from_numpy(masks['upper']) == 1)[0]
    lower_idx = torch.where(torch.from_numpy(masks['lower']) == 1)[0]

    for i in range(13):
        idx = upper_idx[i*3]
        full_pose[:, idx:idx+3] = pose_upper[:, i*3:(i+1)*3]

    full_pose[:, 75:165] = pose_hands  # joints 25-54

    for i in range(9):
        idx = lower_idx[i*3]
        full_pose[:, idx:idx+3] = pose_lower[:, i*3:(i+1)*3]

    return full_pose.reshape(bs, n, 165)


def main():
    print("="*60)
    print("GestureLSM MeanFlow Inference")
    print("="*60)
    print(f"Device: {device}")

    # Config
    audio_path = 'demo/examples/2_scott_0_1_1.wav'
    npz_path = 'demo/examples/2_scott_0_1_1.npz'
    output_path = 'output_motion.npz'

    # Load config
    sys.argv = ['', '-c', 'configs/meanflow_rvqvae_128_hf.yaml']
    args, cfg = config.parse_args()

    # Load normalization
    mean = np.load(args.mean_pose_path)
    std = np.load(args.std_pose_path)
    norm = {
        'mean_upper': torch.from_numpy(mean[upper_body_mask]).float().to(device),
        'std_upper': torch.from_numpy(std[upper_body_mask]).float().to(device),
        'mean_hands': torch.from_numpy(mean[hands_body_mask]).float().to(device),
        'std_hands': torch.from_numpy(std[hands_body_mask]).float().to(device),
        'mean_lower': torch.from_numpy(mean[lower_body_mask]).float().to(device),
        'std_lower': torch.from_numpy(std[lower_body_mask]).float().to(device),
        'trans_mean': torch.from_numpy(np.load(args.mean_trans_path)).float().to(device),
        'trans_std': torch.from_numpy(np.load(args.std_trans_path)).float().to(device),
    }

    # Load VQ-VAEs
    print("\n1. Loading VQ-VAE models...")
    vq_models = load_vqvae_models(args)

    # Load generator
    print("\n2. Loading MeanFlow generator...")
    model_module = __import__(f'models.{cfg.model.model_name}', fromlist=['something'])
    model = getattr(model_module, cfg.model.g_name)(cfg)
    model = torch.nn.DataParallel(model, [0]).to(device)
    other_tools.load_checkpoints(model, args.test_ckpt, args.g_name)
    model.eval()

    # Process audio
    print("\n3. Processing audio...")
    audio_onset, duration = process_audio_to_onset(audio_path)
    print(f"   Duration: {duration:.2f}s, Features: {audio_onset.shape}")

    # Prepare seed
    print("\n4. Preparing seed latents...")
    seed_latent, masks = prepare_seed_latent(npz_path, vq_models, norm, args)
    print(f"   Seed shape: {seed_latent.shape}")

    # Run inference
    print("\n5. Running MeanFlow inference...")
    out_upper, out_hands, out_lower, trans = run_inference(
        model, vq_models, audio_onset, seed_latent, norm, args
    )

    # Convert to poses
    print("\n6. Converting to axis-angle poses...")
    poses = convert_to_poses(out_upper, out_hands, out_lower, masks)

    # Load ground truth for betas/expressions
    gt = np.load(npz_path, allow_pickle=True)
    n_out = poses.shape[1]

    # Save
    print("\n7. Saving output...")
    np.savez(output_path,
        betas=gt['betas'],
        poses=poses.squeeze(0).cpu().numpy(),
        expressions=gt['expressions'][:n_out] if n_out <= len(gt['expressions']) else np.tile(gt['expressions'][-1], (n_out, 1)),
        trans=trans.squeeze(0).cpu().numpy(),
        model='smplx2020',
        gender='neutral',
        mocap_frame_rate=30,
    )

    print(f"\n{'='*60}")
    print(f"SUCCESS! Saved to: {output_path}")
    print(f"  Frames: {n_out}")
    print(f"  Poses: {poses.shape}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
