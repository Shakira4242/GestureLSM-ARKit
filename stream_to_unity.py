#!/usr/bin/env python3
"""
Stream generated motion to Unity via UDP.
Proper denormalization with angle wrapping.
"""
import os
import argparse
import socket
import time
import numpy as np

UNITY_HOST = '127.0.0.1'
UNITY_PORT = 7777

UNITY_TO_OUR = {
    0: 1, 1: 5, 2: 6, 3: 9, 4: 10, 5: 11, 6: 12,
    7: 13, 8: 14, 9: 15, 10: 17, 11: 18, 12: 19, 13: 20,
    14: 22, 15: 23, 16: 24, 17: 25, 18: 27, 19: 28, 20: 29, 21: 30,
    22: 32, 23: 33, 24: 34, 25: 36, 26: 37, 27: 38, 28: 39,
    29: 40, 30: 41, 31: 42, 32: 44, 33: 45, 34: 46, 35: 47,
    36: 49, 37: 50, 38: 51, 39: 52, 40: 54, 41: 55, 42: 56, 43: 57,
    44: 59, 45: 60, 46: 61,
}


def wrap_angle(angle):
    """Wrap angle to -180 to 180 range."""
    return ((angle + 180) % 360) - 180


def continuous_wrap(angles):
    """
    Wrap angles while maintaining continuity across frames.
    Prevents sudden jumps when crossing ±180° boundary.

    angles: (n_frames, n_dims) array
    """
    result = np.zeros_like(angles)
    result[0] = np.array([wrap_angle(a) for a in angles[0]])

    for i in range(1, len(angles)):
        for j in range(angles.shape[1]):
            current = angles[i, j]
            prev = result[i-1, j]

            # Find the wrapped value closest to previous
            wrapped = wrap_angle(current)

            # Check if adding/subtracting 360 gets us closer
            candidates = [wrapped, wrapped + 360, wrapped - 360]
            distances = [abs(c - prev) for c in candidates]
            result[i, j] = candidates[np.argmin(distances)]

    return result


def euler_to_quaternion(x, y, z, order='ZXY'):
    """
    Convert Euler angles (degrees) to quaternion (x, y, z, w).

    order: rotation order used in BVH (common: ZXY, ZYX, XYZ)
    BVH typically uses ZXY for most joints.
    """
    x, y, z = np.radians([x, y, z])

    # Half angles
    cx, sx = np.cos(x/2), np.sin(x/2)
    cy, sy = np.cos(y/2), np.sin(y/2)
    cz, sz = np.cos(z/2), np.sin(z/2)

    # Build quaternions for each axis
    qx = np.array([sx, 0, 0, cx])  # rotation around X
    qy = np.array([0, sy, 0, cy])  # rotation around Y
    qz = np.array([0, 0, sz, cz])  # rotation around Z

    def quat_mult(q1, q2):
        """Quaternion multiplication (x, y, z, w format)."""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ], dtype=np.float32)

    axis_quats = {'X': qx, 'Y': qy, 'Z': qz}

    # Apply rotations in BVH order (right to left for local/intrinsic)
    # BVH uses extrinsic convention, so we multiply left to right
    result = np.array([0, 0, 0, 1], dtype=np.float32)  # identity
    for axis in order:
        result = quat_mult(axis_quats[axis], result)

    return result


def convert_body_to_unity(body_frame, rotation_order='ZXY'):
    """Convert 75-joint body to Unity's 47-joint quaternion format."""
    rotations = body_frame.reshape(75, 3)
    unity_quats = np.zeros((47, 4), dtype=np.float32)
    unity_quats[:, 3] = 1.0

    for unity_idx, our_idx in UNITY_TO_OUR.items():
        if our_idx < 75:
            # Data should already be continuously wrapped
            x, y, z = rotations[our_idx]
            unity_quats[unity_idx] = euler_to_quaternion(x, y, z, rotation_order)

    return unity_quats.flatten()


def interpolate_frames(data, target_frames):
    src_frames = len(data)
    if src_frames == target_frames:
        return data
    indices = np.linspace(0, src_frames - 1, target_frames)
    result = np.zeros((target_frames, data.shape[1]), dtype=data.dtype)
    for i, idx in enumerate(indices):
        low = int(idx)
        high = min(low + 1, src_frames - 1)
        frac = idx - low
        result[i] = data[low] * (1 - frac) + data[high] * frac
    return result


def stream_to_unity(body, face, audio=None, audio_sr=16000, fps=30):
    import sounddevice as sd

    # Apply continuous wrapping to prevent angle discontinuities
    print("Applying continuous angle wrapping...")
    body_wrapped = continuous_wrap(body)
    print(f"Body range after continuous wrap: {body_wrapped.min():.1f} to {body_wrapped.max():.1f}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    n_frames = len(body_wrapped)
    frame_time = 1.0 / fps

    print(f"Streaming {n_frames} frames at {fps} FPS...")
    
    if audio is not None:
        sd.play(audio, audio_sr)
    
    start_time = time.perf_counter()

    for i in range(n_frames):
        body_quats = convert_body_to_unity(body_wrapped[i])
        face_frame = np.clip(face[i], 0, 1).astype(np.float32)
        packet = np.concatenate([body_quats, face_frame]).astype(np.float32)
        sock.sendto(packet.tobytes(), (UNITY_HOST, UNITY_PORT))

        target_time = start_time + (i + 1) * frame_time
        sleep_time = target_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)

    if audio is not None:
        sd.wait()
    
    sock.close()
    print("Done!")


def generate_and_stream(audio_path, checkpoint, stats_dir, ddim_steps=25, fps=30):
    from train_diffsheg_style import TransformerDenoiser, DiffusionTrainer
    import torch
    import librosa

    device = torch.device('cpu')
    print(f"Loading model...")
    model = TransformerDenoiser(motion_dim=276, audio_dim=128, hidden_dim=512,
                                 num_layers=8, num_heads=8, dropout=0.1)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    stats = {
        'body_mean': np.load(os.path.join(stats_dir, 'body_mean.npy')),
        'body_std': np.load(os.path.join(stats_dir, 'body_std.npy')),
        'face_mean': np.load(os.path.join(stats_dir, 'face_mean.npy')),
        'face_std': np.load(os.path.join(stats_dir, 'face_std.npy')),
    }

    print(f"Processing audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio) / sr
    
    audio_18k = librosa.resample(audio, orig_sr=16000, target_sr=18000)
    mel = librosa.feature.melspectrogram(y=audio_18k, sr=18000, hop_length=1200, n_mels=128)
    mel = np.swapaxes(mel[..., :-1], -1, -2).astype(np.float32)
    mel_tensor = torch.from_numpy(mel).unsqueeze(0)

    print(f"Generating ({ddim_steps} DDIM steps)...")
    sampler = DiffusionTrainer(model, timesteps=1000, device=device)
    with torch.no_grad():
        motion = sampler.ddim_sample(mel_tensor, num_steps=ddim_steps)
    motion = motion.squeeze(0).numpy()

    # PROPER denormalization
    body = motion[:, :225] * stats['body_std'] + stats['body_mean']
    face = motion[:, 225:] * stats['face_std'] + stats['face_mean']
    
    print(f"Body range (raw denorm): {body.min():.1f} to {body.max():.1f}")
    print(f"Face range: {face.min():.2f} to {face.max():.2f}")

    # Interpolate to match audio duration
    target_frames = int(duration * fps)
    print(f"Interpolating {len(body)} -> {target_frames} frames")
    
    body = interpolate_frames(body, target_frames)
    face = interpolate_frames(face, target_frames)

    stream_to_unity(body, face, audio=audio, audio_sr=16000, fps=fps)


def test_saved_motion(motion_dir, rotation_order='ZXY'):
    """Test with pre-saved motion files to debug."""
    body = np.load(os.path.join(motion_dir, 'body.npy'))
    face = np.load(os.path.join(motion_dir, 'face.npy'))

    print(f"\n=== DEBUG: Testing saved motion ===")
    print(f"Body shape: {body.shape}")
    print(f"Body range (raw): {body.min():.1f} to {body.max():.1f}")

    # Apply continuous wrapping
    body_wrapped = continuous_wrap(body)
    print(f"Body range (continuous wrap): {body_wrapped.min():.1f} to {body_wrapped.max():.1f}")

    # Check RForeArm before/after continuous wrap
    rforearm_raw = body[:, 33:36]
    rforearm_wrapped = body_wrapped[:, 33:36]
    print(f"\n--- RForeArm X-rotation (first 5 frames) ---")
    print(f"Raw:      {[f'{v:.1f}' for v in rforearm_raw[:5, 0]]}")
    print(f"Cont.wrap:{[f'{v:.1f}' for v in rforearm_wrapped[:5, 0]]}")

    # Check frame-to-frame change after continuous wrap
    diff_raw = np.diff(rforearm_raw, axis=0)
    diff_wrapped = np.diff(rforearm_wrapped, axis=0)
    print(f"\nRForeArm frame-to-frame change:")
    print(f"  Raw:     mean={np.abs(diff_raw).mean():.1f}, max={np.abs(diff_raw).max():.1f}")
    print(f"  Wrapped: mean={np.abs(diff_wrapped).mean():.1f}, max={np.abs(diff_wrapped).max():.1f}")

    # Check a few joints with continuous wrapping
    joint_names = ["Spine", "Neck", "RShoulder", "RArm", "RForeArm", "RHand"]
    our_indices = [1, 5, 9, 10, 11, 12]

    print(f"\n--- Frame 0 joint rotations (after continuous wrap) ---")
    rotations = body_wrapped[0].reshape(75, 3)
    for name, o_idx in zip(joint_names, our_indices):
        angles = rotations[o_idx]
        quat = euler_to_quaternion(angles[0], angles[1], angles[2], rotation_order)
        print(f"{name:12s}: ({angles[0]:7.1f}, {angles[1]:7.1f}, {angles[2]:7.1f}) "
              f"quat=({quat[0]:6.3f}, {quat[1]:6.3f}, {quat[2]:6.3f}, {quat[3]:6.3f})")


def main():
    parser = argparse.ArgumentParser(description='Stream motion to Unity')
    parser.add_argument('--audio', help='Input WAV file')
    parser.add_argument('--checkpoint', default='./outputs/diffsheg_style/model_final.pth')
    parser.add_argument('--stats_dir', default='./outputs/diffsheg_style/')
    parser.add_argument('--ddim_steps', type=int, default=25)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--test', action='store_true', help='Test with saved motion')
    parser.add_argument('--motion_dir', default='./outputs/generated/', help='Dir with body.npy/face.npy')
    parser.add_argument('--rotation_order', default='ZXY', help='BVH rotation order (ZXY, ZYX, XYZ)')

    args = parser.parse_args()

    print("=" * 50)
    print(f"Unity Streaming -> {UNITY_HOST}:{UNITY_PORT}")
    print("=" * 50)

    if args.test:
        test_saved_motion(args.motion_dir, args.rotation_order)
    elif args.audio:
        generate_and_stream(args.audio, args.checkpoint, args.stats_dir, args.ddim_steps, args.fps)
    else:
        print("Provide --audio or --test")


if __name__ == "__main__":
    main()
