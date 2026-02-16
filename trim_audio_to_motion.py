#!/usr/bin/env python3
"""
Trim audio to match motion duration.
Usage: python trim_audio_to_motion.py --npz sanity_test_ep60.npz --audio path/to/audio.wav
"""
import argparse
import numpy as np
import librosa
import soundfile as sf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', required=True, help='Motion NPZ file')
    parser.add_argument('--audio', required=True, help='Audio WAV file')
    parser.add_argument('--output', help='Output audio path (default: <npz_name>_audio.wav)')
    args = parser.parse_args()

    # Get motion duration
    data = np.load(args.npz)
    motion_frames = len(data['body'])
    fps = int(data['fps']) if 'fps' in data.files else 30
    duration = motion_frames / fps
    print(f"Motion: {motion_frames} frames at {fps}fps = {duration:.1f}s")

    # Load and trim audio
    audio, sr = librosa.load(args.audio, sr=16000)
    audio_duration = len(audio) / sr
    print(f"Audio: {audio_duration:.1f}s")

    audio_trimmed = audio[:int(duration * sr)]
    print(f"Trimmed to: {duration:.1f}s")

    # Save
    output_path = args.output or args.npz.replace('.npz', '_audio.wav')
    sf.write(output_path, audio_trimmed, sr)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
