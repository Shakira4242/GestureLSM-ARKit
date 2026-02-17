#!/usr/bin/env python3
"""
Train VQ-VAE for BVH + ARKit format with multi-GPU support.
Body: 225 dims (75 joints × 3 Euler angles, normalized to -180..180)
Face: 51 dims (ARKit blendshapes)

Usage:
    # Single GPU
    python train_vqvae_bvh.py --body_part body --epochs 100

    # Multi-GPU (4x H100)
    torchrun --nproc_per_node=4 train_vqvae_bvh.py --body_part body --epochs 100 --ddp
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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

    # Multi-GPU / DDP
    parser.add_argument('--ddp', action='store_true',
                        help='Enable DistributedDataParallel for multi-GPU training')

    return parser.parse_args()


class Args:
    """Wrapper to pass to dataloader."""
    pass


def setup_ddp():
    """Initialize distributed training."""
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    args = get_args()

    # DDP setup
    if args.ddp:
        rank, world_size, local_rank = setup_ddp()
        device = torch.device(f'cuda:{local_rank}')
        is_main = (rank == 0)
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_main = True

    if is_main:
        print(f"Device: {device}")
        if args.ddp:
            print(f"DDP enabled: {world_size} GPUs")

    # Output directory
    out_dir = os.path.join(args.out_dir, args.body_part)
    if is_main:
        os.makedirs(out_dir, exist_ok=True)

    if args.ddp:
        dist.barrier()  # Wait for rank 0 to create directory

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
    if is_main:
        print("Loading dataset...")
    train_dataset = BEATNormalizedDataset(data_args, split='train')

    # DataLoader with optional DistributedSampler
    if args.ddp:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   sampler=sampler, num_workers=args.num_workers,
                                   drop_last=True, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers,
                                   drop_last=True, pin_memory=True)

    if is_main:
        print(f"Dataset: {len(train_dataset)} samples")
        if args.ddp:
            print(f"Batch size per GPU: {args.batch_size}, Total: {args.batch_size * world_size}")

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

    if is_main:
        print(f"Training {args.body_part}: {dim_pose} dims")

    # Create VQ-VAE model
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

    if is_main:
        print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Wrap model in DDP
    if args.ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        model_unwrapped = model.module
    else:
        model_unwrapped = model

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
        if is_main:
            print(f"Loading checkpoint from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        model_unwrapped.load_state_dict(checkpoint['net'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_loss = checkpoint.get('loss', float('inf'))
        if is_main:
            print(f"Resumed from epoch {checkpoint.get('epoch', '?')}, best_loss={best_loss:.4f}")
            print(f"Continuing from epoch {start_epoch} to {args.epochs}")
    elif args.resume or args.resume_from:
        if is_main:
            print(f"No checkpoint found to resume from, starting fresh")

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()

        # Set epoch for distributed sampler
        if args.ddp:
            train_loader.sampler.set_epoch(epoch)

        total_loss = 0
        total_recon = 0
        total_commit = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not is_main)
        for batch in pbar:
            motion = batch['motion'].to(device)  # (B, T, 276)

            # Extract the part we're training
            motion = motion[:, :, slice_start:slice_end]  # (B, T, dim_pose)

            # Forward
            output = model(motion)
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

            if is_main:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                })

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_commit = total_commit / len(train_loader)

        if is_main:
            print(f"Epoch {epoch}: loss={avg_loss:.4f}, recon={avg_recon:.4f}, commit={avg_commit:.4f}")

        # Save checkpoint (only on main process)
        if is_main:
            if epoch % args.save_every == 0 or avg_loss < best_loss:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_path = os.path.join(out_dir, 'best.pth')
                else:
                    save_path = os.path.join(out_dir, f'epoch_{epoch}.pth')

                torch.save({
                    'epoch': epoch,
                    'net': model_unwrapped.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': avg_loss,
                    'args': vars(args),
                }, save_path)
                print(f"Saved: {save_path}")

    # Save final (only if it doesn't exist - never overwrite)
    if is_main:
        final_path = os.path.join(out_dir, 'final.pth')
        if not os.path.exists(final_path):
            torch.save({
                'epoch': args.epochs,
                'net': model_unwrapped.state_dict(),
                'loss': avg_loss,
                'args': vars(args),
            }, final_path)
            print(f"Saved: {final_path}")
        else:
            print(f"Skipping final.pth save - already exists")

        # Save normalization stats for inference (only if they don't exist - never overwrite)
        stats = train_dataset.get_norm_stats()
        stats_files = [
            ('body_mean.npy', stats['body_mean']),
            ('body_std.npy', stats['body_std']),
            ('face_mean.npy', stats['face_mean']),
            ('face_std.npy', stats['face_std']),
        ]
        for fname, data in stats_files:
            fpath = os.path.join(out_dir, fname)
            if not os.path.exists(fpath):
                np.save(fpath, data)
                print(f"Saved: {fpath}")
            else:
                print(f"Skipping {fname} - already exists")

        print("Done!")

    # Cleanup
    if args.ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
