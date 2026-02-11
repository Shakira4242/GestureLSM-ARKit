#!/usr/bin/env python3
"""
MeanFlow Trainer for GestureLSM with ARKit blendshapes support.
Based on: https://arxiv.org/abs/2505.13447 (Mean Flows for One-step Generative Modeling)
Reference implementation: https://github.com/noamelata/MeanFlow
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloaders.build_vocab import Vocab
sys.modules['__main__'].Vocab = Vocab

from utils import config, other_tools
from utils import rotation_conversions as rc
from utils.joints import upper_body_mask, hands_body_mask, lower_body_mask
from dataloaders.data_tools import joints_list
from models.vq.model import RVQVAE

logger = logging.getLogger(__name__)

# Custom Face VQ-VAE (same architecture as train_face_vq_arkit.py)
class FaceVQVAE(nn.Module):
    def __init__(self, input_dim=51, hidden_dim=256, latent_dim=128, num_codes=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 4, 2, 1), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Conv1d(hidden_dim, latent_dim, 4, 2, 1), nn.BatchNorm1d(latent_dim), nn.ReLU()
        )
        self.codebook = nn.Embedding(num_codes, latent_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, hidden_dim, 4, 2, 1), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, input_dim, 4, 2, 1)
        )

    def map2latent(self, x):
        """Encode to latent (for generator training). x: (B, T, D) -> (B, T', latent)"""
        z = self.encoder(x.permute(0, 2, 1))  # (B, latent, T')
        return z.permute(0, 2, 1)  # (B, T', latent)

    def latent2origin(self, z):
        """Decode from latent. z: (B, T', latent) -> (B, T, D)"""
        x = self.decoder(z.permute(0, 2, 1))  # (B, D, T)
        return x.permute(0, 2, 1), None  # (B, T, D)

#################################################################################
#                           MeanFlow Loss Functions                             #
#################################################################################

def meanflow_loss(model, x_data, noise, t, r, seed, audio_features,
                  p_mean=-0.4, p_std=1.0, adaptive_weight_p=1.0):
    """
    Compute MeanFlow loss based on the average velocity identity.

    The key identity: u(z_t, r, t) = v - (t-r) * du/dt
    where v = noise - data (true velocity)

    Args:
        model: The denoiser network that predicts u(z, t, r)
        x_data: Clean data (latents)
        noise: Gaussian noise
        t: End time (batch,)
        r: Start time (batch,), r <= t
        seed: Seed vectors for conditioning
        audio_features: Audio conditioning features
        p_mean: Mean for logit-normal time sampling
        p_std: Std for logit-normal time sampling
        adaptive_weight_p: Exponent for adaptive loss weighting

    Returns:
        loss: Scalar loss value
        loss_dict: Dictionary with loss components
    """
    b = x_data.shape[0]
    device = x_data.device

    # Reshape t, r for broadcasting: (b,) -> (b, 1, 1, 1)
    t_bc = t.view(b, 1, 1, 1)
    r_bc = r.view(b, 1, 1, 1)

    # Interpolate: z_t = (1-t)*data + t*noise (linear path)
    z_t = (1 - t_bc) * x_data + t_bc * noise

    # True velocity: v = noise - data
    v = noise - x_data

    # Define forward function for JVP
    def forward_fn(z, t_in, r_in):
        return model(
            x=z,
            timesteps=t_in,
            cond_time=r_in,
            seed=seed,
            at_feat=audio_features,
        )

    # Compute JVP: tangents are (dz/dt, dt/dt, dr/dt) = (v, 1, 0)
    tangents = (v, torch.ones(b, device=device), torch.zeros(b, device=device))

    # u = model output, dudt = time derivative via JVP
    u, dudt = torch.autograd.functional.jvp(
        forward_fn,
        (z_t, t, r),
        tangents,
        create_graph=True
    )

    # Target from MeanFlow identity: u_tgt = v - (t-r) * du/dt
    # Stop gradient on dudt to prevent higher-order optimization
    u_tgt = v - (t_bc - r_bc) * dudt.detach()

    # MSE loss
    loss_per_sample = (u - u_tgt).pow(2).sum(dim=(1, 2, 3))

    # Adaptive weighting (optional, helps with training stability)
    if adaptive_weight_p > 0:
        weights = 1.0 / (loss_per_sample.detach() + 1e-3).pow(adaptive_weight_p)
        loss = (loss_per_sample * weights).mean()
    else:
        loss = loss_per_sample.mean()

    loss_dict = {
        'meanflow_loss': loss.item(),
        'loss_unweighted': loss_per_sample.mean().item(),
    }

    return loss, loss_dict


def sample_time_meanflow(batch_size, device, p_mean=-0.4, p_std=1.0, rate_same=0.25):
    """
    Sample (r, t) pairs for MeanFlow training.

    Uses logit-normal distribution and ensures r <= t.
    With probability rate_same, r = t (data proportion for boundary condition).

    Args:
        batch_size: Number of samples
        device: torch device
        p_mean: Mean for logit-normal
        p_std: Std for logit-normal
        rate_same: Probability that r = t

    Returns:
        r: Start times (batch,)
        t: End times (batch,), t >= r
    """
    # Sample two times from logit-normal
    raw = torch.randn((2, batch_size), device=device)

    # With probability rate_same, make them the same
    same_mask = torch.rand((1, batch_size), device=device) < rate_same
    raw = torch.where(same_mask, raw[:1].repeat(2, 1), raw)

    # Transform to (0, 1) via logit-normal
    times = raw.mul(p_std).add(p_mean).sigmoid()

    # Sort to ensure r <= t
    r, t = times.sort(dim=0).values.unbind(dim=0)

    return r, t


#################################################################################
#                              Trainer Class                                    #
#################################################################################

class MeanFlowTrainer:
    """
    MeanFlow trainer for GestureLSM gesture generation.
    Supports ARKit 51-blendshape face output.
    """

    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # MeanFlow hyperparameters
        self.p_mean = cfg.model.get('time_mu', -0.4)
        self.p_std = cfg.model.get('time_sigma', 1.0)
        self.rate_same = cfg.model.get('data_proportion', 0.25)
        self.adaptive_p = cfg.model.get('adaptive_p', 1.0)

        # Whether to use face (ARKit blendshapes)
        self.use_face = cfg.model.get('use_exp', False)
        self.facial_dims = args.facial_dims if hasattr(args, 'facial_dims') else 100

        logger.info(f"MeanFlow Trainer initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Use face: {self.use_face} (dims: {self.facial_dims})")
        logger.info(f"  Time sampling: p_mean={self.p_mean}, p_std={self.p_std}")
        logger.info(f"  Data proportion (rate_same): {self.rate_same}")

    def load_vq_models(self):
        """Load VQ-VAE models for body parts (and face if enabled)."""
        vq_args = type('Args', (), {
            'mu': 0.99, 'nb_code': 1024, 'code_dim': 128,
            'down_t': 2, 'stride_t': 2, 'width': 512, 'depth': 3,
            'dilation_growth_rate': 3, 'vq_act': 'relu', 'vq_norm': None,
            'num_quantizers': 6, 'shared_codebook': False,
            'quantize_dropout_prob': 0.2, 'quantize_dropout_cutoff_index': 0
        })()

        configs = {
            'upper': (78, self.args.vqvae_upper_path),
            'hands': (180, self.args.vqvae_hands_path),
            'lower': (57, self.args.vqvae_lower_path),
        }

        # Add face VQ if enabled
        if self.use_face:
            configs['face'] = (self.facial_dims, self.args.vqvae_face_path)

        self.vq_models = {}
        for name, (dim, path) in configs.items():
            logger.info(f"Loading VQ-VAE: {name} (dim={dim}) from {path}")
            if name == 'face':
                # Use custom FaceVQVAE for face model
                model = FaceVQVAE(input_dim=dim)
            else:
                # Use RVQVAE for body parts
                model = RVQVAE(vq_args, dim, vq_args.nb_code, vq_args.code_dim, vq_args.code_dim,
                              vq_args.down_t, vq_args.stride_t, vq_args.width, vq_args.depth,
                              vq_args.dilation_growth_rate, vq_args.vq_act, vq_args.vq_norm)
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt['net'])
            model.eval().to(self.device)
            self.vq_models[name] = model

        logger.info(f"Loaded {len(self.vq_models)} VQ-VAE models")

    def load_normalization(self):
        """Load mean/std for normalization."""
        self.norm = {
            'mean_upper': torch.from_numpy(np.load(self.args.mean_pose_path)[upper_body_mask]).float().to(self.device),
            'std_upper': torch.from_numpy(np.load(self.args.std_pose_path)[upper_body_mask]).float().to(self.device),
            'mean_hands': torch.from_numpy(np.load(self.args.mean_pose_path)[hands_body_mask]).float().to(self.device),
            'std_hands': torch.from_numpy(np.load(self.args.std_pose_path)[hands_body_mask]).float().to(self.device),
            'mean_lower': torch.from_numpy(np.load(self.args.mean_pose_path)[lower_body_mask]).float().to(self.device),
            'std_lower': torch.from_numpy(np.load(self.args.std_pose_path)[lower_body_mask]).float().to(self.device),
            'trans_mean': torch.from_numpy(np.load(self.args.mean_trans_path)).float().to(self.device),
            'trans_std': torch.from_numpy(np.load(self.args.std_trans_path)).float().to(self.device),
        }

        # Face normalization (for ARKit, typically no normalization needed as values are 0-1)
        if self.use_face:
            if hasattr(self.args, 'mean_face_path') and os.path.exists(self.args.mean_face_path):
                self.norm['mean_face'] = torch.from_numpy(np.load(self.args.mean_face_path)).float().to(self.device)
                self.norm['std_face'] = torch.from_numpy(np.load(self.args.std_face_path)).float().to(self.device)
            else:
                # ARKit blendshapes are already normalized 0-1
                self.norm['mean_face'] = torch.zeros(self.facial_dims).to(self.device)
                self.norm['std_face'] = torch.ones(self.facial_dims).to(self.device)

        logger.info("Loaded normalization statistics")

    def load_generator(self):
        """Load MeanFlow generator model."""
        model_module = __import__(f'models.{self.cfg.model.model_name}', fromlist=['something'])
        self.model = getattr(model_module, self.cfg.model.g_name)(self.cfg)
        self.model = self.model.to(self.device)

        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"MeanFlow generator: {n_params/1e6:.2f}M parameters")

        return self.model

    def encode_to_latent(self, batch):
        """Encode motion data to VQ latent space."""
        # This would be called during training to get latent targets
        # Implementation depends on your dataloader format
        pass

    def train_step(self, batch):
        """
        Single training step for MeanFlow.

        Args:
            batch: Dictionary containing:
                - latent: Target latent vectors (b, seq_len, latent_dim)
                - audio_features: Encoded audio features
                - seed: Seed vectors

        Returns:
            loss: Scalar loss
            loss_dict: Dictionary with loss components
        """
        latent = batch['latent'].to(self.device)
        audio_features = batch['audio_features'].to(self.device)
        seed = batch['seed'].to(self.device)

        b = latent.shape[0]

        # Sample noise
        noise = torch.randn_like(latent)

        # Sample times
        r, t = sample_time_meanflow(
            b, self.device,
            p_mean=self.p_mean,
            p_std=self.p_std,
            rate_same=self.rate_same
        )

        # Compute MeanFlow loss
        loss, loss_dict = meanflow_loss(
            model=self.model.denoiser,
            x_data=latent,
            noise=noise,
            t=t,
            r=r,
            seed=seed,
            audio_features=audio_features,
            p_mean=self.p_mean,
            p_std=self.p_std,
            adaptive_weight_p=self.adaptive_p,
        )

        return loss, loss_dict

    def train(self, train_loader, num_epochs, lr=1e-4, save_dir='./outputs/meanflow'):
        """
        Full training loop.

        Args:
            train_loader: DataLoader for training data
            num_epochs: Number of epochs
            lr: Learning rate
            save_dir: Directory to save checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0
        )

        self.model.train()
        global_step = 0

        for epoch in range(num_epochs):
            epoch_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch in pbar:
                optimizer.zero_grad()

                loss, loss_dict = self.train_step(batch)

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()

                epoch_loss += loss.item()
                global_step += 1

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

                # Log every 100 steps
                if global_step % 100 == 0:
                    logger.info(f"Step {global_step}: loss={loss.item():.4f}")

            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                ckpt_path = os.path.join(save_dir, f"meanflow_epoch{epoch+1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")

        logger.info("Training complete!")

    def run_training(self, train_loader, num_epochs, lr, save_dir):
        """
        Full training loop with data preprocessing.
        """
        os.makedirs(save_dir, exist_ok=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0
        )

        self.model.train()
        global_step = 0

        for epoch in range(num_epochs):
            epoch_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch in pbar:
                # Extract data from batch
                poses = batch['poses'].to(self.device)  # (B, T, 330)
                face = batch['face'].to(self.device)    # (B, T, 51)
                audio = batch['audio'].to(self.device)  # (B, T_audio, 2)

                b, seq_len = poses.shape[:2]

                # Split poses into body parts and normalize
                upper = poses[:, :, :78]
                hands = poses[:, :, 78:258]
                lower = poses[:, :, 258:]

                # Normalize body parts
                upper = (upper - self.norm['mean_upper']) / (self.norm['std_upper'] + 1e-7)
                hands = (hands - self.norm['mean_hands']) / (self.norm['std_hands'] + 1e-7)
                lower = (lower - self.norm['mean_lower']) / (self.norm['std_lower'] + 1e-7)

                # Encode to latent space
                with torch.no_grad():
                    lat_upper = self.vq_models['upper'].map2latent(upper)
                    lat_hands = self.vq_models['hands'].map2latent(hands)
                    lat_lower = self.vq_models['lower'].map2latent(lower)
                    lat_face = self.vq_models['face'].map2latent(face)

                # Concatenate latents: (B, T', latent_dim * 4)
                latent = torch.cat([lat_upper, lat_hands, lat_lower, lat_face], dim=-1)

                # Get seed (first few frames)
                seed = latent[:, :8, :]

                # Process audio for conditioning
                # Downsample audio to match latent temporal resolution
                audio_len = latent.shape[1]
                audio_ds = F.interpolate(
                    audio.permute(0, 2, 1),
                    size=audio_len,
                    mode='linear'
                ).permute(0, 2, 1)

                # Sample noise
                noise = torch.randn_like(latent)

                # Sample times
                r, t = sample_time_meanflow(
                    b, self.device,
                    p_mean=self.p_mean,
                    p_std=self.p_std,
                    rate_same=self.rate_same
                )

                optimizer.zero_grad()

                # Forward pass through model
                t_bc = t.view(b, 1, 1)
                r_bc = r.view(b, 1, 1)
                z_t = (1 - t_bc) * latent + t_bc * noise
                v = noise - latent

                # Model prediction
                pred = self.model(
                    in_audio=audio_ds,
                    pre_seq=seed,
                    in_facial=None,
                    in_id=None,
                    in_word=None,
                    in_emo=None,
                    t=t,
                    r=r,
                    z_t=z_t
                )

                # Simple MSE loss (simplified MeanFlow)
                loss = F.mse_loss(pred, v)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                global_step += 1

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
                ckpt_path = os.path.join(save_dir, f"meanflow_epoch{epoch+1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")

        logger.info("Training complete!")


#################################################################################
#                                  Main                                         #
#################################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='MeanFlow Training for GestureLSM')
    parser.add_argument('-c', '--config', default='configs/meanflow_arkit.yaml', help='Config file')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--save_dir', default='./outputs/meanflow_arkit', help='Save directory')

    cli_args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load config
    sys.argv = ['', '-c', cli_args.config]
    args, cfg = config.parse_args()

    # Create trainer
    trainer = MeanFlowTrainer(args, cfg)

    # Load models
    trainer.load_vq_models()
    trainer.load_normalization()
    trainer.load_generator()

    # Create dataloader using BEATArkitDataset
    from dataloaders.beat_arkit import BEATArkitDataset

    dataset = BEATArkitDataset(args, split='train')
    train_loader = DataLoader(
        dataset,
        batch_size=cli_args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    print("="*60)
    print("MeanFlow Training Starting!")
    print("="*60)
    print(f"Config: {cli_args.config}")
    print(f"Use face (ARKit): {trainer.use_face}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {cli_args.batch_size}")
    print(f"Epochs: {cli_args.epochs}")
    print("="*60)

    # Run training
    trainer.run_training(train_loader, cli_args.epochs, cli_args.lr, cli_args.save_dir)
