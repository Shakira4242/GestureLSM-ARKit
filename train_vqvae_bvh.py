#!/usr/bin/env python3
"""
Train VQ-VAE for BVH + ARKit format.
Body: 225 dims (75 joints × 3 Euler angles, normalized to -180..180)
Face: 51 dims (ARKit blendshapes)

Usage:
    python train_vqvae_bvh.py --body_part body --epochs 100
    python train_vqvae_bvh.py --body_part face --epochs 100
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from models.vq.model import RVQVAE
from dataloaders.beat_normalized import BEATNormalizedDataset


def get_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_path', type=str,
                        default='./datasets/BEAT/beat_english_v0.2.1/beat_english_v0.2.1/')
    parser.add_argument('--cache_path', type=str,
                        default='./datasets/beat_cache/beat_bvh_arkit/')
    parser.add_argument('--speakers', type=int, nargs='+', default=[2])
    parser.add_argument('--pose_length', type=int, default=64)
    parser.add_argument('--new_cache', action='store_true')

    # What to train
    parser.add_argument('--body_part', type=str, default='body',
                        choices=['body', 'face', 'all'])

    # VQ-VAE architecture
    parser.add_argument('--code_dim', type=int, default=128)
    parser.add_argument('--nb_code', type=int, default=1024)
    parser.add_argument('--down_t', type=int, default=2)
    parser.add_argument('--stride_t', type=int, default=2)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--dilation_growth_rate', type=int, default=3)
    parser.add_argument('--vq_act', type=str, default='relu')
    parser.add_argument('--vq_norm', type=str, default=None)

    # Training
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-5)  # Lower LR for stable training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--commit_weight', type=float, default=0.02)
    parser.add_argument('--vel_weight', type=float, default=0.1)

    # Output
    parser.add_argument('--out_dir', type=str, default='./outputs/vqvae_bvh/')
    parser.add_argument('--save_every', type=int, default=20)

    # Performance
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Dataloader workers (set to ~vCPU/3)')

    # Resume training
    parser.add_argument('--resume', action='store_true',
                        help='Auto-resume from best.pth if it exists')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Specific checkpoint path to resume from')

    return parser.parse_args()


class Args:
    """Wrapper to pass to dataloader."""
    pass


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Output directory
    out_dir = os.path.join(args.out_dir, args.body_part)
    os.makedirs(out_dir, exist_ok=True)

    # Create dataloader args
    data_args = Args()
    data_args.data_path = args.data_path
    data_args.cache_path = args.cache_path
    data_args.pose_fps = 30
    data_args.audio_sr = 16000
    data_args.pose_length = args.pose_length
    data_args.stride = 10
    data_args.training_speakers = args.speakers
    data_args.new_cache = args.new_cache

    # Load dataset
    print("Loading dataset...")
    train_dataset = BEATNormalizedDataset(data_args, split='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers,
                               drop_last=True, pin_memory=True)

    print(f"Dataset: {len(train_dataset)} samples")

    # Determine input dimension based on body part
    if args.body_part == 'body':
        dim_pose = 225  # 75 joints × 3
        slice_start, slice_end = 0, 225
    elif args.body_part == 'face':
        dim_pose = 51   # ARKit blendshapes
        slice_start, slice_end = 225, 276
    else:  # all
        dim_pose = 276  # body + face
        slice_start, slice_end = 0, 276

    print(f"Training {args.body_part}: {dim_pose} dims")

    # Create VQ-VAE model
    # Create a simple args namespace for RVQVAE
    vq_args = Args()
    vq_args.num_quantizers = 6
    vq_args.shared_codebook = False
    vq_args.quantize_dropout_prob = 0.2
    vq_args.quantize_dropout_cutoff_index = 0
    vq_args.mu = 0.99

    model = RVQVAE(
        vq_args,
        dim_pose,
        args.nb_code,
        args.code_dim,
        args.code_dim,
        args.down_t,
        args.stride_t,
        args.width,
        args.depth,
        args.dilation_growth_rate,
        args.vq_act,
        args.vq_norm
    ).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Loss
    recon_loss_fn = nn.SmoothL1Loss()

    # Resume from checkpoint if requested or if checkpoint exists
    start_epoch = 1
    best_loss = float('inf')

    # Determine checkpoint path
    resume_path = None
    if args.resume_from:
        resume_path = args.resume_from
    elif args.resume:
        # Auto-find best checkpoint
        best_path = os.path.join(out_dir, 'best.pth')
        if os.path.exists(best_path):
            resume_path = best_path

    if resume_path and os.path.exists(resume_path):
        print(f"Loading checkpoint from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['net'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_loss = checkpoint.get('loss', float('inf'))
        print(f"Resumed from epoch {checkpoint.get('epoch', '?')}, best_loss={best_loss:.4f}")
        print(f"Continuing from epoch {start_epoch} to {args.epochs}")
    elif args.resume or args.resume_from:
        print(f"No checkpoint found to resume from, starting fresh")

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0
        total_recon = 0
        total_commit = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            motion = batch['motion'].to(device)  # (B, T, 276)

            # Extract the part we're training
            motion = motion[:, :, slice_start:slice_end]  # (B, T, dim_pose)

            # Forward
            output = model(motion)
            # Decoder already outputs (B, T, dim) - see encdec.py line 128
            pred_motion = output['rec_pose']
            commit_loss = output['commit_loss']

            # Reconstruction loss
            recon_loss = recon_loss_fn(pred_motion, motion)

            # Velocity loss (smoothness)
            vel_gt = motion[:, 1:] - motion[:, :-1]
            vel_pred = pred_motion[:, 1:] - pred_motion[:, :-1]
            vel_loss = recon_loss_fn(vel_pred, vel_gt)

            # Total loss
            loss = recon_loss + args.commit_weight * commit_loss + args.vel_weight * vel_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_commit += commit_loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
            })

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_commit = total_commit / len(train_loader)

        print(f"Epoch {epoch}: loss={avg_loss:.4f}, recon={avg_recon:.4f}, commit={avg_commit:.4f}")

        # Save checkpoint
        if epoch % args.save_every == 0 or avg_loss < best_loss:
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_path = os.path.join(out_dir, 'best.pth')
            else:
                save_path = os.path.join(out_dir, f'epoch_{epoch}.pth')

            torch.save({
                'epoch': epoch,
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
                'args': vars(args),
            }, save_path)
            print(f"Saved: {save_path}")

    # Save final
    torch.save({
        'epoch': args.epochs,
        'net': model.state_dict(),
        'loss': avg_loss,
        'args': vars(args),
    }, os.path.join(out_dir, 'final.pth'))

    # Save normalization stats for inference
    stats = train_dataset.get_norm_stats()
    np.save(os.path.join(out_dir, 'body_mean.npy'), stats['body_mean'])
    np.save(os.path.join(out_dir, 'body_std.npy'), stats['body_std'])
    np.save(os.path.join(out_dir, 'face_mean.npy'), stats['face_mean'])
    np.save(os.path.join(out_dir, 'face_std.npy'), stats['face_std'])
    print(f"Saved normalization stats to {out_dir}")

    print("Done!")


if __name__ == "__main__":
    main()
