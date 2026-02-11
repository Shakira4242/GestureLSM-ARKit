#!/usr/bin/env python3
"""
Train Face VQ-VAE for ARKit 51 blendshapes.
This must be trained BEFORE training MeanFlow with face support.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#################################################################################
#                           Face VQ-VAE Model                                   #
#################################################################################

class VQVAEConv1D(nn.Module):
    """
    VQ-VAE for facial blendshapes using 1D convolutions.
    Based on the existing VQVAEConvZero but simplified for face.
    """

    def __init__(self, input_dim=51, hidden_dim=256, latent_dim=128,
                 num_codes=512, num_layers=2, seq_len=128):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_codes = num_codes

        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else latent_dim
            encoder_layers.extend([
                nn.Conv1d(in_dim, out_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
            ])
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Codebook
        self.codebook = nn.Embedding(num_codes, latent_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else input_dim
            decoder_layers.extend([
                nn.ConvTranspose1d(in_dim, out_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(out_dim) if i < num_layers - 1 else nn.Identity(),
                nn.ReLU(inplace=True) if i < num_layers - 1 else nn.Identity(),
            ])
            in_dim = out_dim
        self.decoder = nn.Sequential(*decoder_layers)

        self.commitment_cost = 0.25

    def encode(self, x):
        """Encode input to latent space. x: (B, T, D) -> (B, latent_dim, T')"""
        x = x.permute(0, 2, 1)  # (B, D, T)
        z = self.encoder(x)     # (B, latent_dim, T')
        return z

    def quantize(self, z):
        """Vector quantization. z: (B, latent_dim, T')"""
        B, D, T = z.shape

        # Reshape for distance computation
        z_flat = z.permute(0, 2, 1).reshape(-1, D)  # (B*T', D)

        # Compute distances to codebook
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=1)
        )

        # Find nearest codes
        indices = distances.argmin(dim=1)  # (B*T',)
        z_q = self.codebook(indices)       # (B*T', D)

        # Reshape back
        z_q = z_q.view(B, T, D).permute(0, 2, 1)  # (B, D, T')

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, indices.view(B, T)

    def decode(self, z_q):
        """Decode quantized latent to output. z_q: (B, latent_dim, T') -> (B, T, D)"""
        x_recon = self.decoder(z_q)          # (B, D, T)
        x_recon = x_recon.permute(0, 2, 1)   # (B, T, D)
        return x_recon

    def forward(self, x):
        """Full forward pass."""
        z = self.encode(x)
        z_q, indices = self.quantize(z)
        x_recon = self.decode(z_q)
        return x_recon, z, z_q, indices

    def compute_loss(self, x, x_recon, z, z_q):
        """Compute VQ-VAE loss."""
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)

        # Codebook loss (move codebook towards encoder output)
        codebook_loss = F.mse_loss(z_q.detach(), z)

        # Commitment loss (keep encoder output close to codebook)
        commitment_loss = F.mse_loss(z_q, z.detach())

        total_loss = recon_loss + codebook_loss + self.commitment_cost * commitment_loss

        return {
            'total': total_loss,
            'recon': recon_loss,
            'codebook': codebook_loss,
            'commitment': commitment_loss,
        }

    def map2latent(self, x):
        """Encode to continuous latent (for generator training)."""
        z = self.encode(x)
        return z.permute(0, 2, 1)  # (B, T', latent_dim)

    def latent2origin(self, z):
        """Decode from continuous latent."""
        z = z.permute(0, 2, 1)  # (B, latent_dim, T')
        x_recon = self.decode(z)
        return x_recon, None


#################################################################################
#                           Dataset                                             #
#################################################################################

class FaceArkitDataset(Dataset):
    """Dataset for ARKit face blendshapes from BEAT JSON files."""

    def __init__(self, data_path, speaker_ids=[2], seq_len=128, stride=20):
        self.seq_len = seq_len
        self.samples = []

        for speaker_id in speaker_ids:
            speaker_dir = os.path.join(data_path, str(speaker_id))
            if not os.path.exists(speaker_dir):
                logger.warning(f"Speaker dir not found: {speaker_dir}")
                continue

            json_files = [f for f in os.listdir(speaker_dir) if f.endswith('.json')]
            logger.info(f"Speaker {speaker_id}: {len(json_files)} JSON files")

            for json_file in json_files:
                json_path = os.path.join(speaker_dir, json_file)
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    # Extract blendshape weights (60 FPS)
                    frames = [frame['weights'] for frame in data['frames']]
                    face_60fps = np.array(frames, dtype=np.float32)

                    # Resample to 30 FPS
                    face = face_60fps[::2]

                    # Slice into windows
                    for start in range(0, len(face) - seq_len, stride):
                        self.samples.append(face[start:start + seq_len])

                except Exception as e:
                    logger.warning(f"Error loading {json_file}: {e}")

        logger.info(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx])


#################################################################################
#                           Training                                            #
#################################################################################

def train_face_vq(args):
    """Train Face VQ-VAE for ARKit blendshapes."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Dataset
    dataset = FaceArkitDataset(
        data_path=args.data_path,
        speaker_ids=args.speakers,
        seq_len=args.seq_len,
        stride=args.stride,
    )

    if len(dataset) == 0:
        logger.error("No data found! Check data_path.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Model
    model = VQVAEConv1D(
        input_dim=51,  # ARKit blendshapes
        hidden_dim=256,
        latent_dim=128,
        num_codes=512,
        num_layers=2,
        seq_len=args.seq_len,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params / 1e6:.2f}M")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_recon = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch = batch.to(device)

            optimizer.zero_grad()

            x_recon, z, z_q, indices = model(batch)
            losses = model.compute_loss(batch, x_recon, z, z_q)

            losses['total'].backward()
            optimizer.step()

            total_loss += losses['total'].item()
            total_recon += losses['recon'].item()

            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'recon': f"{losses['recon'].item():.4f}",
            })

        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}, recon={avg_recon:.4f}")

        # Save checkpoint
        if (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.save_dir, f'face_arkit_51_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            logger.info(f"Saved: {ckpt_path}")

    # Save final model
    final_path = os.path.join(args.save_dir, 'face_arkit_51.pth')
    torch.save({'net': model.state_dict()}, final_path)
    logger.info(f"Final model saved: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Face VQ-VAE for ARKit')
    parser.add_argument('--data_path', default='./datasets/BEAT/beat_english_v0.2.1/beat_english_v0.2.1/',
                        help='Path to BEAT dataset with JSON files')
    parser.add_argument('--speakers', nargs='+', type=int, default=[2],
                        help='Speaker IDs to use')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')
    parser.add_argument('--stride', type=int, default=20, help='Stride for slicing')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_dir', default='./ckpt', help='Save directory')

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("Face VQ-VAE Training for ARKit 51 Blendshapes")
    logger.info("="*60)
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Speakers: {args.speakers}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info("="*60)

    train_face_vq(args)
