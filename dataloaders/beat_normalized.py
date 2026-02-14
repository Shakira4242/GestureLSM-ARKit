"""
BEAT Dataset loader with AXIS-ANGLE conversion (like DiffSHEG).

KEY FIX: Convert Euler angles → axis-angle BEFORE training.
- Euler angles can accumulate (e.g., 37°, 397°, -2482° are the same pose)
- Axis-angle is bounded (~±π radians) and has ONE representation per rotation
- This ensures consistent training targets

Output: 276 dims = 225 body (75 joints × 3 axis-angle) + 51 face (ARKit)
"""
import os
import json
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from loguru import logger
import librosa

# Import rotation converter (from DiffSHEG)
from dataloaders.rotation_converter import euler_angles_to_axis_angle

# All 75 BEAT joints (full body including hands and legs)
BEAT_JOINTS = [
    "Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Neck1", "Head", "HeadEnd",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4",
    "RightHandRing", "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4",
    "RightHandPinky", "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4",
    "RightHandIndex", "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumb4",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4",
    "LeftHandRing", "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4",
    "LeftHandPinky", "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4",
    "LeftHandIndex", "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumb4",
    "RightUpLeg", "RightLeg", "RightFoot", "RightForeFoot", "RightToeBase", "RightToeBaseEnd",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftForeFoot", "LeftToeBase", "LeftToeBaseEnd",
]

# ARKit 51 blendshape names
ARKIT_BLENDSHAPES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"
]


def normalize_angle(angle):
    """Normalize angle to -180..180 range."""
    return ((angle + 180) % 360) - 180


def normalize_angles_continuous(angles):
    """
    Normalize angles to -180..180 while maintaining temporal continuity.
    This prevents discontinuities at the ±180° boundary within a sequence.

    angles: (n_frames, n_dims) array of Euler angles in degrees
    """
    result = np.zeros_like(angles)

    # First frame: simple normalization
    result[0] = np.array([normalize_angle(a) for a in angles[0]])

    # Subsequent frames: choose the wrapped value closest to previous
    for i in range(1, len(angles)):
        for j in range(angles.shape[1]):
            current = angles[i, j]
            prev = result[i-1, j]

            # Normalize to -180..180
            wrapped = normalize_angle(current)

            # Check if ±360 gets us closer to previous frame
            candidates = [wrapped, wrapped + 360, wrapped - 360]
            distances = [abs(c - prev) for c in candidates]
            result[i, j] = candidates[np.argmin(distances)]

    return result


def parse_bvh(filepath):
    """Parse BVH file and extract joint rotations."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    joints = []
    joint_channels = {}
    frames = []
    frame_time = 0.033333
    in_hierarchy = True
    current_joint = None

    for line in lines:
        line = line.strip()
        if line == "MOTION":
            in_hierarchy = False
            continue

        if in_hierarchy:
            if line.startswith("ROOT") or line.startswith("JOINT"):
                joint_name = line.split()[1]
                joints.append(joint_name)
                current_joint = joint_name
            elif line.startswith("CHANNELS") and current_joint:
                parts = line.split()
                num_channels = int(parts[1])
                joint_channels[current_joint] = parts[2:2+num_channels]
        else:
            if line.startswith("Frame Time:"):
                frame_time = float(line.split(":")[1].strip())
            elif line and not line.startswith("Frames"):
                try:
                    values = [float(v) for v in line.split()]
                    if len(values) > 10:
                        frames.append(values)
                except:
                    pass

    return joints, joint_channels, frames, frame_time


def extract_joint_rotations(joints, joint_channels, frames, target_joints):
    """Extract rotations for target joints from BVH frames."""
    n_frames = len(frames)
    n_joints = len(target_joints)
    rotations = np.zeros((n_frames, n_joints, 3), dtype=np.float32)

    # Build offset map
    joint_offsets = {}
    offset = 0
    for joint in joints:
        joint_offsets[joint] = offset
        offset += len(joint_channels.get(joint, []))

    # Map BVH joint names to our target joints
    bvh_to_target = {}
    for i, target in enumerate(target_joints):
        if target in joints:
            bvh_to_target[target] = i
        else:
            # Try common variations
            variations = [
                target,
                target.replace("Middle", "Mid"),
                target.replace("Index", "Idx"),
                target.replace("Pinky", "Pink"),
                target.replace("Ring", "Rng"),
                target.replace("Thumb", "Thb"),
            ]
            for var in variations:
                if var in joints:
                    bvh_to_target[var] = i
                    break

    for frame_idx, frame_data in enumerate(frames):
        for bvh_joint, target_idx in bvh_to_target.items():
            if bvh_joint not in joint_channels:
                continue

            channels = joint_channels[bvh_joint]
            offset = joint_offsets[bvh_joint]

            rot = [0.0, 0.0, 0.0]
            for ci, ch in enumerate(channels):
                if offset + ci < len(frame_data):
                    if 'Xrotation' in ch:
                        rot[0] = frame_data[offset + ci]
                    elif 'Yrotation' in ch:
                        rot[1] = frame_data[offset + ci]
                    elif 'Zrotation' in ch:
                        rot[2] = frame_data[offset + ci]

            rotations[frame_idx, target_idx] = rot

    return rotations


class BEATNormalizedDataset(Dataset):
    """
    BEAT dataset with PROPERLY NORMALIZED Euler angles.

    Key difference from beat_raw_arkit.py:
    - Euler angles are normalized to -180..180 BEFORE computing mean/std
    - This ensures consistent representation and bounded statistics
    """

    def __init__(self, args, split='train'):
        self.args = args
        self.split = split

        self.data_path = args.data_path
        self.pose_fps = getattr(args, 'pose_fps', 30)
        self.audio_sr = getattr(args, 'audio_sr', 16000)
        self.pose_length = getattr(args, 'pose_length', 34)
        self.stride = getattr(args, 'stride', 10)
        self.speakers = getattr(args, 'training_speakers', [2])

        # Output dimensions
        self.body_dim = 225  # 75 joints × 3 axis-angle (radians)
        self.face_dim = 51   # ARKit blendshapes
        self.total_dim = self.body_dim + self.face_dim  # 276

        # Load or build cache
        self.cache_path = getattr(args, 'cache_path', './datasets/beat_cache/beat_normalized/')
        os.makedirs(self.cache_path, exist_ok=True)

        cache_file = os.path.join(self.cache_path, f'{split}_cache.pkl')
        new_cache = getattr(args, 'new_cache', False)

        if os.path.exists(cache_file) and not new_cache:
            logger.info(f"Loading cache from {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.samples = cache_data['samples']
                self.body_mean = cache_data['body_mean']
                self.body_std = cache_data['body_std']
                self.face_mean = cache_data['face_mean']
                self.face_std = cache_data['face_std']
        else:
            logger.info(f"Building cache for {split} split...")
            self.samples, stats = self._build_cache()
            self.body_mean, self.body_std = stats['body_mean'], stats['body_std']
            self.face_mean, self.face_std = stats['face_mean'], stats['face_std']

            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'samples': self.samples,
                    'body_mean': self.body_mean,
                    'body_std': self.body_std,
                    'face_mean': self.face_mean,
                    'face_std': self.face_std,
                }, f)
            logger.info(f"Saved cache to {cache_file}")

        # Log statistics to verify they're reasonable
        logger.info(f"Dataset: {len(self.samples)} samples")
        logger.info(f"Body mean range: [{self.body_mean.min():.1f}, {self.body_mean.max():.1f}]")
        logger.info(f"Body std range: [{self.body_std.min():.1f}, {self.body_std.max():.1f}]")
        logger.info(f"Face mean range: [{self.face_mean.min():.3f}, {self.face_mean.max():.3f}]")
        logger.info(f"Face std range: [{self.face_std.min():.3f}, {self.face_std.max():.3f}]")

    def _build_cache(self):
        """Build dataset cache with properly normalized angles."""
        samples = []
        all_body = []
        all_face = []

        for speaker_id in self.speakers:
            speaker_dir = os.path.join(self.data_path, str(speaker_id))
            if not os.path.exists(speaker_dir):
                logger.warning(f"Speaker directory not found: {speaker_dir}")
                continue

            bvh_files = sorted([f for f in os.listdir(speaker_dir) if f.endswith('.bvh')])
            logger.info(f"Speaker {speaker_id}: found {len(bvh_files)} BVH files")

            for bvh_file in bvh_files:
                base_name = bvh_file.replace('.bvh', '')

                bvh_path = os.path.join(speaker_dir, bvh_file)
                json_path = os.path.join(speaker_dir, base_name + '.json')
                wav_path = os.path.join(speaker_dir, base_name + '.wav')

                if not os.path.exists(json_path) or not os.path.exists(wav_path):
                    continue

                try:
                    # Load body from BVH
                    joints, joint_channels, frames, frame_time = parse_bvh(bvh_path)
                    bvh_fps = 1.0 / frame_time

                    # Extract rotations
                    rotations = extract_joint_rotations(joints, joint_channels, frames, BEAT_JOINTS)

                    # Resample to target FPS if needed
                    if abs(bvh_fps - self.pose_fps) > 1:
                        ratio = bvh_fps / self.pose_fps
                        indices = np.arange(0, len(rotations), ratio).astype(int)
                        indices = indices[indices < len(rotations)]
                        rotations = rotations[indices]

                    # *** KEY FIX: Convert Euler → axis-angle (like DiffSHEG) ***
                    # rotations shape: (T, 75, 3) in degrees

                    # Step 1: Convert degrees to radians
                    euler_radians = np.deg2rad(rotations)  # (T, 75, 3)

                    # Step 2: Convert to axis-angle using DiffSHEG's convention
                    # DiffSHEG uses "XYZ" convention
                    euler_tensor = torch.from_numpy(euler_radians).float()
                    axis_angle_tensor = euler_angles_to_axis_angle(euler_tensor, "XYZ")
                    body = axis_angle_tensor.numpy()  # (T, 75, 3)

                    # Flatten to (T, 225)
                    body = body.reshape(len(body), -1)

                    # Load face from JSON
                    with open(json_path, 'r') as f:
                        face_data = json.load(f)

                    face_frames = [frame['weights'] for frame in face_data['frames']]
                    face_60fps = np.array(face_frames, dtype=np.float32)

                    # Resample face to pose_fps
                    face_ratio = 60 / self.pose_fps
                    face_indices = np.arange(0, len(face_60fps), face_ratio).astype(int)
                    face_indices = face_indices[face_indices < len(face_60fps)]
                    face = face_60fps[face_indices]

                    # Align lengths
                    min_len = min(len(body), len(face))
                    body = body[:min_len]
                    face = face[:min_len]

                    # Load audio and compute mel
                    audio, sr = librosa.load(wav_path, sr=self.audio_sr)
                    audio_18k = librosa.resample(audio, orig_sr=self.audio_sr, target_sr=18000)
                    mel = librosa.feature.melspectrogram(
                        y=audio_18k, sr=18000, hop_length=1200, n_mels=128
                    )
                    mel = np.swapaxes(mel[..., :-1], -1, -2).astype(np.float32)

                    # Slice into windows
                    for start in range(0, min_len - self.pose_length, self.stride):
                        end = start + self.pose_length

                        body_slice = body[start:end].astype(np.float32)
                        face_slice = face[start:end].astype(np.float32)
                        mel_slice = mel[start:end] if end <= len(mel) else np.zeros((self.pose_length, 128), dtype=np.float32)

                        if len(mel_slice) < self.pose_length:
                            mel_slice = np.pad(mel_slice, ((0, self.pose_length - len(mel_slice)), (0, 0)))

                        samples.append({
                            'body': body_slice,
                            'face': face_slice,
                            'mel': mel_slice,
                        })

                        all_body.append(body_slice)
                        all_face.append(face_slice)

                except Exception as e:
                    logger.warning(f"Error processing {base_name}: {e}")
                    continue

        # Compute normalization stats on NORMALIZED angles
        if all_body:
            all_body = np.concatenate(all_body, axis=0)
            all_face = np.concatenate(all_face, axis=0)

            body_mean = all_body.mean(axis=0)
            body_std = all_body.std(axis=0) + 1e-7
            face_mean = all_face.mean(axis=0)
            face_std = all_face.std(axis=0) + 1e-7

            # Verify statistics are reasonable
            logger.info(f"Body angles after normalization: min={all_body.min():.1f}, max={all_body.max():.1f}")
            logger.info(f"Body mean: min={body_mean.min():.1f}, max={body_mean.max():.1f}")
            logger.info(f"Body std: min={body_std.min():.1f}, max={body_std.max():.1f}")
        else:
            body_mean = np.zeros(self.body_dim)
            body_std = np.ones(self.body_dim)
            face_mean = np.zeros(self.face_dim)
            face_std = np.ones(self.face_dim)

        stats = {
            'body_mean': body_mean,
            'body_std': body_std,
            'face_mean': face_mean,
            'face_std': face_std,
        }

        return samples, stats

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Z-score normalization
        body = (sample['body'] - self.body_mean) / self.body_std
        face = (sample['face'] - self.face_mean) / self.face_std

        # Concatenate
        motion = np.concatenate([body, face], axis=-1)

        return {
            'motion': torch.from_numpy(motion.astype(np.float32)),
            'mel': torch.from_numpy(sample['mel'].astype(np.float32)),
        }

    def get_norm_stats(self):
        """Return normalization stats for inference."""
        return {
            'body_mean': self.body_mean,
            'body_std': self.body_std,
            'face_mean': self.face_mean,
            'face_std': self.face_std,
        }


if __name__ == "__main__":
    # Test the dataloader
    class Args:
        data_path = './datasets/BEAT/beat_english_v0.2.1/beat_english_v0.2.1/'
        cache_path = './datasets/beat_cache/beat_normalized_test/'
        pose_fps = 30
        audio_sr = 16000
        pose_length = 34
        stride = 10
        training_speakers = [2]
        new_cache = True

    args = Args()
    dataset = BEATNormalizedDataset(args, split='train')

    print(f"\nDataset size: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Motion shape: {sample['motion'].shape}")
        print(f"Mel shape: {sample['mel'].shape}")

        # Verify normalized motion is reasonable
        motion = sample['motion'].numpy()
        print(f"Normalized motion range: [{motion.min():.2f}, {motion.max():.2f}]")
