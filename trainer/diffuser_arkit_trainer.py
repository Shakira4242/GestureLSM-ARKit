"""
Diffuser trainer with ARKit 51-blendshape face support.
Extends the base generative trainer to use custom FaceVQVAE.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from trainer.generative_trainer import CustomTrainer as BaseCustomTrainer
from models.vq.model import RVQVAE


# Custom Face VQ-VAE (same architecture as train_face_vq_arkit.py)
class FaceVQVAE(nn.Module):
    """VQ-VAE for ARKit 51 blendshapes."""

    def __init__(self, input_dim=51, hidden_dim=256, latent_dim=128, num_codes=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 4, 2, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, latent_dim, 4, 2, 1),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )
        self.codebook = nn.Embedding(num_codes, latent_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, hidden_dim, 4, 2, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, input_dim, 4, 2, 1),
        )

    def map2latent(self, x):
        """Encode to latent. x: (B, T, D) -> (B, T', latent)"""
        z = self.encoder(x.permute(0, 2, 1))  # (B, latent, T')
        return z.permute(0, 2, 1)  # (B, T', latent)

    def latent2origin(self, z):
        """Decode from latent. z: (B, T', latent) -> (B, T, D)"""
        x = self.decoder(z.permute(0, 2, 1))  # (B, D, T)
        return x.permute(0, 2, 1), None  # (B, T, D)


class CustomTrainer(BaseCustomTrainer):
    """
    Diffuser trainer with ARKit face support.
    """

    def __init__(self, cfg, args):
        # Call parent init (but we'll override VQ model loading)
        super().__init__(cfg, args)

        # Check if ARKit face is enabled
        self.use_arkit_face = getattr(cfg, 'use_arkit_face', False) or \
                              getattr(cfg.model, 'use_arkit_face', False)

        if self.use_arkit_face:
            self._load_arkit_face_vq()

    def _load_arkit_face_vq(self):
        """Load the ARKit Face VQ-VAE model."""
        face_vq_path = getattr(self.cfg, 'vqvae_face_path', './ckpt/face_arkit_51.pth')

        if not os.path.exists(face_vq_path):
            logger.warning(f"Face VQ-VAE not found at {face_vq_path}")
            logger.warning("Train it first with: python train_face_vq_arkit.py")
            self.vq_model_face = None
            return

        logger.info(f"Loading ARKit Face VQ-VAE from {face_vq_path}")

        self.vq_model_face = FaceVQVAE(input_dim=51)
        ckpt = torch.load(face_vq_path, map_location='cpu', weights_only=False)
        self.vq_model_face.load_state_dict(ckpt['net'])
        self.vq_model_face.eval().to(self.rank)

        logger.info("ARKit Face VQ-VAE loaded successfully (51 blendshapes)")

    def encode_motion(self, tar_pose, tar_trans=None, tar_face=None):
        """
        Encode motion to latent space using VQ-VAE models.

        Args:
            tar_pose: (B, T, 330) body pose in rot6d
            tar_trans: (B, T, 3) translation (optional)
            tar_face: (B, T, 51) ARKit blendshapes (optional)

        Returns:
            latents: Dictionary of latent codes for each body part
        """
        latents = {}

        # Split body pose
        tar_upper = tar_pose[:, :, :78]
        tar_hands = tar_pose[:, :, 78:258]
        tar_lower = tar_pose[:, :, 258:]

        # Encode body parts
        with torch.no_grad():
            latents['upper'] = self.vq_model_upper.map2latent(tar_upper)
            latents['hands'] = self.vq_model_hands.map2latent(tar_hands)
            latents['lower'] = self.vq_model_lower.map2latent(tar_lower)

            # Encode face if available
            if tar_face is not None and self.vq_model_face is not None:
                latents['face'] = self.vq_model_face.map2latent(tar_face)

        return latents

    def decode_motion(self, latents):
        """
        Decode latent codes back to motion.

        Args:
            latents: Dictionary of latent codes

        Returns:
            tar_pose: (B, T, 330) body pose
            tar_face: (B, T, 51) ARKit blendshapes (if available)
        """
        with torch.no_grad():
            upper, _ = self.vq_model_upper.latent2origin(latents['upper'])
            hands, _ = self.vq_model_hands.latent2origin(latents['hands'])
            lower, _ = self.vq_model_lower.latent2origin(latents['lower'])

            tar_pose = torch.cat([upper, hands, lower], dim=-1)

            tar_face = None
            if 'face' in latents and self.vq_model_face is not None:
                tar_face, _ = self.vq_model_face.latent2origin(latents['face'])

        return tar_pose, tar_face

    def train(self, epoch):
        """Training loop with ARKit face support."""
        self.model.train()

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
            self.opt.zero_grad()

            # Get data from batch
            tar_pose = batch['poses'].cuda()  # (B, T, 330)
            in_audio = batch['audio'].cuda()  # (B, T_audio, 2)

            # Get face if available (ARKit blendshapes)
            tar_face = batch.get('face', None)
            if tar_face is not None:
                tar_face = tar_face.cuda()  # (B, T, 51)

            # Encode to latent space
            latents = self.encode_motion(tar_pose, tar_face=tar_face)

            # Concatenate latents
            latent_list = [latents['upper'], latents['hands'], latents['lower']]
            if 'face' in latents:
                latent_list.append(latents['face'])
            combined_latent = torch.cat(latent_list, dim=-1)

            # Get seed frames
            seed = combined_latent[:, :8, :]

            # Sample timestep
            t = torch.randint(
                0, self.model.module.scheduler.config.num_train_timesteps,
                (combined_latent.shape[0],), device=combined_latent.device
            )

            # Add noise
            noise = torch.randn_like(combined_latent)
            noisy_latent = self.model.module.scheduler.add_noise(combined_latent, noise, t)

            # Predict noise
            pred = self.model(
                in_audio=in_audio,
                pre_seq=seed,
                noisy_x=noisy_latent,
                timesteps=t,
            )

            # Compute loss
            loss = F.mse_loss(pred, noise)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

            self.tracker.update("predict_x0_loss", loss.item())

        self.opt_s.step()
        self.tracker.log(epoch, self.rank)
