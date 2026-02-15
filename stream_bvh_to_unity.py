#!/usr/bin/env python3
"""
Stream BVH (axis-angle) + ARKit motion to Unity via UDP.

Input: NPZ file with:
  - body: (N, 225) - 75 joints × 3 axis-angle (radians)
  - face: (N, 51) - 51 ARKit blendshapes

Output: UDP packets to Unity (port 7777)
  - 47 joints × 4 quaternion + 51 face = 239 floats per frame

Usage:
    python stream_bvh_to_unity.py --npz sanity_test_ep20.npz
    python stream_bvh_to_unity.py --npz sanity_test_ep20.npz --audio test.wav
"""
import argparse
import socket
import time
import numpy as np

UNITY_HOST = '127.0.0.1'
UNITY_PORT = 7777

# Maps Unity's 47 joint indices to our 75-joint BVH indices
# Unity idx -> BVH idx
UNITY_TO_BVH = {
    0: 1,   # Spine
    1: 5,   # Neck
    2: 6,   # Neck1/Head
    3: 9,   # RShoulder
    4: 10,  # RArm
    5: 11,  # RArm1/RForeArm
    6: 12,  # RHand
    7: 13,  # RHandM1
    8: 14,  # RHandM2
    9: 15,  # RHandM3
    10: 17, # RHandR1
    11: 18, # RHandR2
    12: 19, # RHandR3
    13: 20, # RHandR (metacarpal)
    14: 22, # RHandP1
    15: 23, # RHandP2
    16: 24, # RHandP3
    17: 25, # RHandP (metacarpal)
    18: 27, # RHandI1
    19: 28, # RHandI2
    20: 29, # RHandI3
    21: 30, # RHandI (metacarpal)
    22: 32, # RHandT1
    23: 33, # RHandT2
    24: 34, # RHandT3
    25: 36, # LShoulder
    26: 37, # LArm
    27: 38, # LArm1/LForeArm
    28: 39, # LHand
    29: 40, # LHandM1
    30: 41, # LHandM2
    31: 42, # LHandM3
    32: 44, # LHandR1
    33: 45, # LHandR2
    34: 46, # LHandR3
    35: 47, # LHandR (metacarpal)
    36: 49, # LHandP1
    37: 50, # LHandP2
    38: 51, # LHandP3
    39: 52, # LHandP (metacarpal)
    40: 54, # LHandI1
    41: 55, # LHandI2
    42: 56, # LHandI3
    43: 57, # LHandI (metacarpal)
    44: 59, # LHandT1
    45: 60, # LHandT2
    46: 61, # LHandT3
}


def axis_angle_to_quaternion(axis_angle):
    """
    Convert axis-angle (3D vector) to quaternion (x, y, z, w).

    The axis-angle representation encodes rotation as:
    - Direction of vector = rotation axis
    - Magnitude of vector = rotation angle (radians)

    Args:
        axis_angle: (3,) array [ax, ay, az] in radians

    Returns:
        (4,) array [qx, qy, qz, qw] quaternion
    """
    angle = np.linalg.norm(axis_angle)

    if angle < 1e-8:
        # No rotation - return identity quaternion
        return np.array([0, 0, 0, 1], dtype=np.float32)

    # Normalize to get axis
    axis = axis_angle / angle

    # Convert to quaternion
    half_angle = angle / 2
    sin_half = np.sin(half_angle)
    cos_half = np.cos(half_angle)

    qx = axis[0] * sin_half
    qy = axis[1] * sin_half
    qz = axis[2] * sin_half
    qw = cos_half

    return np.array([qx, qy, qz, qw], dtype=np.float32)


def convert_body_to_unity(body_frame, coord_fix=1):
    """
    Convert 75-joint axis-angle body to Unity's 47-joint quaternion format.

    Args:
        body_frame: (225,) array - 75 joints × 3 axis-angle values
        coord_fix: Coordinate system fix (1 = negate X and Z for Unity)

    Returns:
        (188,) array - 47 joints × 4 quaternion values
    """
    rotations = body_frame.reshape(75, 3)
    unity_quats = np.zeros((47, 4), dtype=np.float32)
    unity_quats[:, 3] = 1.0  # Default to identity quaternion

    for unity_idx, bvh_idx in UNITY_TO_BVH.items():
        if bvh_idx < 75:
            axis_angle = rotations[bvh_idx].copy()

            # Coordinate system conversion (BVH Y-up right-handed -> Unity Y-up left-handed)
            if coord_fix == 1:
                axis_angle[0] = -axis_angle[0]  # Negate X
                axis_angle[2] = -axis_angle[2]  # Negate Z

            quat = axis_angle_to_quaternion(axis_angle)
            unity_quats[unity_idx] = quat

    return unity_quats.flatten()


def stream_to_unity(body, face, audio=None, audio_sr=16000, fps=30):
    """
    Stream motion data to Unity via UDP.

    Args:
        body: (N, 225) array - body motion (75 joints × 3 axis-angle)
        face: (N, 51) array - face blendshapes (51 ARKit)
        audio: Optional audio array to play
        audio_sr: Audio sample rate
        fps: Frames per second
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    n_frames = len(body)
    frame_time = 1.0 / fps

    print(f"Streaming {n_frames} frames at {fps} FPS to {UNITY_HOST}:{UNITY_PORT}")
    print(f"Duration: {n_frames / fps:.1f}s")
    print(f"Body shape: {body.shape}, Face shape: {face.shape}")

    # Start audio playback if provided
    if audio is not None:
        try:
            import sounddevice as sd
            sd.play(audio, audio_sr)
            print("Playing audio...")
        except ImportError:
            print("Warning: sounddevice not installed, skipping audio playback")
            audio = None

    start_time = time.perf_counter()

    for i in range(n_frames):
        # Convert body to Unity format (47 joints × 4 quaternions)
        body_quats = convert_body_to_unity(body[i])

        # Clip face blendshapes to 0-1 range
        face_frame = np.clip(face[i], 0, 1).astype(np.float32)

        # Combine into single packet: 188 body + 51 face = 239 floats
        packet = np.concatenate([body_quats, face_frame]).astype(np.float32)

        # Send UDP packet
        sock.sendto(packet.tobytes(), (UNITY_HOST, UNITY_PORT))

        # Progress update every 100 frames
        if i % 100 == 0:
            print(f"  Frame {i}/{n_frames} ({100*i/n_frames:.1f}%)")

        # Timing control
        target_time = start_time + (i + 1) * frame_time
        sleep_time = target_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Wait for audio to finish
    if audio is not None:
        try:
            import sounddevice as sd
            sd.wait()
        except:
            pass

    sock.close()
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Stream BVH motion to Unity')
    parser.add_argument('--npz', required=True, help='Input NPZ file with body and face arrays')
    parser.add_argument('--audio', help='Optional audio file to play alongside')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--host', default='127.0.0.1', help='Unity host')
    parser.add_argument('--port', type=int, default=7777, help='Unity port')
    parser.add_argument('--start_frame', type=int, default=0, help='Start from this frame')
    parser.add_argument('--end_frame', type=int, default=-1, help='End at this frame (-1 = all)')

    args = parser.parse_args()

    global UNITY_HOST, UNITY_PORT
    UNITY_HOST = args.host
    UNITY_PORT = args.port

    print("=" * 60)
    print("BVH + ARKit Streamer to Unity")
    print("=" * 60)

    # Load NPZ
    print(f"Loading: {args.npz}")
    data = np.load(args.npz)
    body = data['body']
    face = data['face']

    # Get FPS from file if available
    if 'fps' in data.files:
        fps = int(data['fps'])
        print(f"Using FPS from file: {fps}")
    else:
        fps = args.fps

    print(f"Body: {body.shape} (75 joints × 3 axis-angle)")
    print(f"Face: {face.shape} (51 ARKit blendshapes)")

    # Slice frames if requested
    start = args.start_frame
    end = args.end_frame if args.end_frame > 0 else len(body)
    body = body[start:end]
    face = face[start:end]
    print(f"Frames: {start} to {end} ({len(body)} total)")

    # Load audio if provided
    audio = None
    audio_sr = 16000
    if args.audio:
        try:
            import librosa
            audio, audio_sr = librosa.load(args.audio, sr=16000)
            print(f"Audio: {args.audio} ({len(audio)/audio_sr:.1f}s)")
        except ImportError:
            print("Warning: librosa not installed, skipping audio")
        except Exception as e:
            print(f"Warning: Could not load audio: {e}")

    print("=" * 60)
    print("Starting stream... (Ctrl+C to stop)")
    print("=" * 60)

    try:
        stream_to_unity(body, face, audio=audio, audio_sr=audio_sr, fps=fps)
    except KeyboardInterrupt:
        print("\nStopped by user")


if __name__ == "__main__":
    main()
