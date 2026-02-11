#!/usr/bin/env python3
"""
Train gesture model with DiffSHEG-compatible output format.
Output: 192 dims = 141 body (47 joints axis-angle) + 51 face (ARKit blendshapes)

This outputs the EXACT same format as DiffSHEG, so it works directly with
your existing Unity receiver.
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


#################################################################################
#                           Simple Diffusion Model                               #
#################################################################################

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class TransformerDenoiser(nn.Module):
    """
    Transformer-based denoiser for gesture generation.
    Takes noisy motion + audio conditioning, outputs denoised motion.
    """

    def __init__(self, motion_dim=192, audio_dim=128, hidden_dim=512,
                 num_layers=8, num_heads=8, dropout=0.1):
        super().__init__()

        self.motion_dim = motion_dim
        self.hidden_dim = hidden_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Motion input projection
        self.motion_proj = nn.Linear(motion_dim, hidden_dim)

        # Audio conditioning
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, motion_dim)

    def forward(self, x, t, audio):
        """
        Args:
            x: Noisy motion (B, T, 192)
            t: Timesteps (B,)
            audio: Mel spectrogram (B, T, 128)

        Returns:
            Predicted noise (B, T, 192)
        """
        B, T, _ = x.shape

        # Time embedding
        t_emb = self.time_embed(t)  # (B, hidden)

        # Project inputs
        x = self.motion_proj(x)  # (B, T, hidden)
        audio = self.audio_proj(audio)  # (B, T, hidden)

        # Add audio conditioning
        x = x + audio

        # Add time embedding (broadcast across sequence)
        x = x + t_emb[:, None, :]

        # Add positional encoding
        x = x + self.pos_embed[:, :T, :]

        # Transformer
        x = self.transformer(x)

        # Output
        x = self.output_proj(x)

        return x


#################################################################################
#                           Diffusion Utilities                                  #
#################################################################################

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionTrainer:
    """Simple diffusion trainer matching DiffSHEG's approach."""

    def __init__(self, model, timesteps=1000, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.timesteps = timesteps

        # Use cosine schedule like DiffSHEG
        betas = cosine_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """Add noise to data (forward diffusion)."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]

        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise, noise

    def p_losses(self, x_start, audio, t=None):
        """Compute training loss."""
        B = x_start.shape[0]

        if t is None:
            t = torch.randint(0, self.timesteps, (B,), device=self.device)

        noise = torch.randn_like(x_start)
        x_noisy, _ = self.q_sample(x_start, t, noise)

        # Predict noise
        noise_pred = self.model(x_noisy, t, audio)

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)

        return loss

    @torch.no_grad()
    def ddim_sample(self, audio, num_steps=25):
        """DDIM sampling (faster than DDPM)."""
        B, T, _ = audio.shape
        device = audio.device

        # Start from noise
        x = torch.randn(B, T, self.model.motion_dim, device=device)

        # DDIM timesteps
        step_size = self.timesteps // num_steps
        timesteps = list(range(0, self.timesteps, step_size))[::-1]

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # Predict noise
            noise_pred = self.model(x, t_batch, audio)

            # DDIM update
            alpha = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0)

            sqrt_alpha = torch.sqrt(alpha)
            sqrt_one_minus_alpha = torch.sqrt(1 - alpha)
            sqrt_alpha_prev = torch.sqrt(alpha_prev)

            # Predict x_0
            x_0_pred = (x - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha

            # DDIM step
            x = sqrt_alpha_prev * x_0_pred + torch.sqrt(1 - alpha_prev) * noise_pred

        return x


#################################################################################
#                           Training Loop                                        #
#################################################################################

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Dataset
    from dataloaders.beat_raw_arkit import BEATRawArkitDataset

    class DataArgs:
        data_path = args.data_path
        cache_path = args.cache_path
        pose_fps = 30
        audio_sr = 16000
        pose_length = 34  # Match DiffSHEG
        stride = 10
        training_speakers = args.speakers
        new_cache = args.new_cache

    dataset = BEATRawArkitDataset(DataArgs(), split='train')

    if len(dataset) == 0:
        logger.error("No data found!")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Save normalization stats
    norm_stats = dataset.get_norm_stats()
    os.makedirs(args.save_dir, exist_ok=True)
    np.save(os.path.join(args.save_dir, 'body_mean.npy'), norm_stats['body_mean'])
    np.save(os.path.join(args.save_dir, 'body_std.npy'), norm_stats['body_std'])
    np.save(os.path.join(args.save_dir, 'face_mean.npy'), norm_stats['face_mean'])
    np.save(os.path.join(args.save_dir, 'face_std.npy'), norm_stats['face_std'])
    logger.info(f"Saved normalization stats to {args.save_dir}")

    # Model
    model = TransformerDenoiser(
        motion_dim=192,  # 141 body + 51 face
        audio_dim=128,   # Mel spectrogram
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params / 1e6:.2f}M")

    # Trainer
    trainer = DiffusionTrainer(model, timesteps=1000, device=device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            motion = batch['motion'].to(device)  # (B, T, 192)
            mel = batch['mel'].to(device)        # (B, T, 128)

            optimizer.zero_grad()
            loss = trainer.p_losses(motion, mel)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        if (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.save_dir, f'model_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    # Save final model
    final_path = os.path.join(args.save_dir, 'model_final.pth')
    torch.save({'model_state_dict': model.state_dict()}, final_path)
    logger.info(f"Training complete! Final model: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DiffSHEG-style gesture model')
    parser.add_argument('--data_path', default='./datasets/BEAT/beat_english_v0.2.1/beat_english_v0.2.1/',
                        help='Path to BEAT dataset')
    parser.add_argument('--cache_path', default='./datasets/beat_cache/beat_raw_arkit/',
                        help='Cache directory')
    parser.add_argument('--speakers', nargs='+', type=int, default=[2],
                        help='Speaker IDs to train on')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_dir', default='./outputs/diffsheg_style/', help='Save directory')
    parser.add_argument('--new_cache', action='store_true', help='Rebuild cache')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("DiffSHEG-Style Training")
    logger.info("Output format: 192 dims (141 body + 51 face)")
    logger.info("=" * 60)
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Speakers: {args.speakers}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info("=" * 60)

    train(args)
