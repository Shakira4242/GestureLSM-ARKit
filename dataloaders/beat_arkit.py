"""
BEAT Dataset loader with ARKit 51-blendshape face support.
Loads facial data from original BEAT JSON files instead of SMPL-X expressions.
"""
import os
import json
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from loguru import logger
import librosa
from numpy.lib import stride_tricks

# ARKit 51 blendshape names (same order as BEAT JSON)
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

assert len(ARKIT_BLENDSHAPES) == 51, f"Expected 51 blendshapes, got {len(ARKIT_BLENDSHAPES)}"


class BEATArkitDataset(Dataset):
    """
    BEAT dataset with ARKit blendshapes for face.

    Combines:
    - Body motion from BEAT2 (SMPL-X NPZ files)
    - Face motion from original BEAT (JSON blendshapes)
    - Audio from BEAT WAV files
    """

    def __init__(self, args, split='train'):
        self.args = args
        self.split = split

        # Paths
        self.beat_json_path = args.data_path  # Original BEAT with JSON
        self.beat2_npz_path = getattr(args, 'data_path_beat2', args.data_path.replace('BEAT/', 'BEAT2/'))

        # Settings
        self.pose_fps = args.pose_fps
        self.audio_sr = args.audio_sr
        self.pose_length = args.pose_length
        self.stride = args.stride
        self.speakers = args.training_speakers

        # Load or build cache
        self.cache_path = args.cache_path
        os.makedirs(self.cache_path, exist_ok=True)

        cache_file = os.path.join(self.cache_path, f'{split}_cache.pkl')
        if os.path.exists(cache_file) and not args.new_cache:
            logger.info(f"Loading cache from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.data = pickle.load(f)
        else:
            logger.info(f"Building cache for {split} split...")
            self.data = self._build_cache()
            with open(cache_file, 'wb') as f:
                pickle.dump(self.data, f)
            logger.info(f"Saved cache to {cache_file}")

        logger.info(f"Dataset: {len(self.data)} samples")

    def _build_cache(self):
        """Build dataset cache by loading and processing all files."""
        data = []

        for speaker_id in self.speakers:
            speaker_dir = os.path.join(self.beat_json_path, str(speaker_id))
            if not os.path.exists(speaker_dir):
                logger.warning(f"Speaker directory not found: {speaker_dir}")
                continue

            # Find all JSON files (face blendshapes)
            json_files = sorted([f for f in os.listdir(speaker_dir) if f.endswith('.json')])
            logger.info(f"Speaker {speaker_id}: found {len(json_files)} files")

            for json_file in json_files:
                base_name = json_file.replace('.json', '')

                # File paths
                json_path = os.path.join(speaker_dir, json_file)
                wav_path = os.path.join(speaker_dir, base_name + '.wav')
                npz_path = self._find_npz_file(speaker_id, base_name)

                # Check all files exist
                if not os.path.exists(wav_path):
                    logger.warning(f"WAV not found: {wav_path}")
                    continue
                if npz_path is None or not os.path.exists(npz_path):
                    logger.warning(f"NPZ not found for {base_name}")
                    continue

                try:
                    # Load data
                    samples = self._process_file(json_path, wav_path, npz_path)
                    data.extend(samples)
                except Exception as e:
                    logger.warning(f"Error processing {base_name}: {e}")
                    continue

        return data

    def _find_npz_file(self, speaker_id, base_name):
        """Find corresponding BEAT2 NPZ file."""
        # Try common paths
        possible_paths = [
            f'./datasets/BEAT2/beat_english_v2.0.0/smplxflame_30/{base_name}.npz',
            os.path.join(self.beat2_npz_path, f'beat_english_v2.0.0/smplxflame_30/{base_name}.npz'),
            os.path.join(self.beat2_npz_path, f'smplxflame_30/{base_name}.npz'),
            os.path.join(self.args.data_path, f'smplxflame_30/{base_name}.npz'),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    def _process_file(self, json_path, wav_path, npz_path):
        """Process a single file and return list of samples."""
        samples = []

        # Load face blendshapes from JSON (60 FPS)
        with open(json_path, 'r') as f:
            face_data = json.load(f)

        face_frames = []
        for frame in face_data['frames']:
            face_frames.append(frame['weights'])
        face_60fps = np.array(face_frames, dtype=np.float32)  # (T_60, 51)

        # Resample to pose_fps (30 FPS)
        stride = 60 // self.pose_fps
        face = face_60fps[::stride]  # (T_30, 51)

        # Load body motion from NPZ
        npz_data = np.load(npz_path, allow_pickle=True)
        poses_aa = npz_data['poses']  # (T, 165) axis-angle
        trans = npz_data['trans']  # (T, 3)

        # Convert axis-angle to rotation 6D (165 -> 330)
        # Each 3-dim axis-angle becomes 6-dim rot6d
        T = poses_aa.shape[0]
        n_joints = poses_aa.shape[1] // 3
        poses_aa_reshaped = poses_aa.reshape(T, n_joints, 3)

        # Convert to rotation matrices then to rot6d
        from scipy.spatial.transform import Rotation
        poses_rot6d = []
        for t in range(T):
            frame_rot6d = []
            for j in range(n_joints):
                aa = poses_aa_reshaped[t, j]
                if np.linalg.norm(aa) < 1e-8:
                    # Identity rotation
                    rot6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
                else:
                    rot_mat = Rotation.from_rotvec(aa).as_matrix()
                    # rot6d is first two columns of rotation matrix
                    rot6d = rot_mat[:, :2].flatten()
                frame_rot6d.append(rot6d)
            poses_rot6d.append(np.concatenate(frame_rot6d))
        poses = np.array(poses_rot6d, dtype=np.float32)  # (T, 330)

        # Align lengths
        min_len = min(len(face), len(poses))
        face = face[:min_len]
        poses = poses[:min_len]
        trans = trans[:min_len]

        # Load and process audio
        audio, sr = librosa.load(wav_path, sr=None)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.audio_sr)

        # Compute audio features (onset + amplitude)
        audio_features = self._compute_audio_features(audio)

        # Slice into windows
        for start in range(0, min_len - self.pose_length, self.stride):
            end = start + self.pose_length

            # Audio slice (account for different FPS)
            audio_start = int(start * self.audio_sr / self.pose_fps)
            audio_end = int(end * self.audio_sr / self.pose_fps)

            # Ensure fixed audio length
            audio_slice = audio_features[audio_start:audio_end]
            target_audio_len = int(self.pose_length * self.audio_sr / self.pose_fps)
            if len(audio_slice) < target_audio_len:
                # Pad if too short
                pad_len = target_audio_len - len(audio_slice)
                audio_slice = np.pad(audio_slice, ((0, pad_len), (0, 0)), mode='edge')
            elif len(audio_slice) > target_audio_len:
                # Truncate if too long
                audio_slice = audio_slice[:target_audio_len]

            sample = {
                'poses': poses[start:end].astype(np.float32),
                'trans': trans[start:end].astype(np.float32),
                'face': face[start:end].astype(np.float32),  # ARKit 51-dim!
                'audio': audio_slice.astype(np.float32),
            }
            samples.append(sample)

        return samples

    def _compute_audio_features(self, audio):
        """Compute onset + amplitude features."""
        frame_length = 1024

        # Amplitude envelope
        shape = (audio.shape[-1] - frame_length + 1, frame_length)
        strides = (audio.strides[-1], audio.strides[-1])
        rolling_view = stride_tricks.as_strided(audio, shape=shape, strides=strides)
        amplitude = np.max(np.abs(rolling_view), axis=1)
        amplitude = np.pad(amplitude, (0, frame_length - 1), mode='constant', constant_values=amplitude[-1])

        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=audio, sr=self.audio_sr, units='frames')
        onset = np.zeros(len(audio), dtype=np.float32)
        onset[onset_frames] = 1.0

        # Combine: [amplitude, onset]
        features = np.stack([amplitude, onset], axis=-1)
        return features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        return {
            'poses': torch.from_numpy(sample['poses']),
            'trans': torch.from_numpy(sample['trans']),
            'face': torch.from_numpy(sample['face']),  # (T, 51) ARKit blendshapes
            'audio': torch.from_numpy(sample['audio']),
        }


def create_dataloader(args, split='train', batch_size=128, num_workers=4):
    """Create dataloader for BEAT ARKit dataset."""
    dataset = BEATArkitDataset(args, split=split)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return loader


if __name__ == "__main__":
    # Test the dataloader
    class Args:
        data_path = './datasets/BEAT/beat_english_v0.2.1/beat_english_v0.2.1/'
        data_path_beat2 = './datasets/BEAT2/beat_english_v2.0.0/'
        cache_path = './datasets/beat_cache/beat_arkit_test/'
        pose_fps = 30
        audio_sr = 16000
        pose_length = 128
        stride = 20
        training_speakers = [2]
        new_cache = True

    args = Args()

    print("Testing BEAT ARKit dataloader...")
    dataset = BEATArkitDataset(args, split='train')
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"  poses shape: {sample['poses'].shape}")
        print(f"  trans shape: {sample['trans'].shape}")
        print(f"  face shape: {sample['face'].shape}")  # Should be (128, 51)
        print(f"  audio shape: {sample['audio'].shape}")
