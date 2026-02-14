import os
import pprint
import random
import sys
import time
import warnings
from typing import Dict

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from dataloaders import data_tools
from dataloaders.data_tools import joints_list
from loguru import logger
from models.vq.model import RVQVAE
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from trainer.base_trainer import BaseTrainer
from utils import (
    data_transfer,
    logger_tools,
    metric,
    other_tools,
    other_tools_hf,
    rotation_conversions as rc,
)
from utils.joints import hands_body_mask, lower_body_mask, upper_body_mask


def convert_15d_to_6d(motion):
    """
    Convert 15D motion to 6D motion, the current motion is 15D, but the eval model is 6D
    """
    bs = motion.shape[0]
    motion_6d = motion.reshape(bs, -1, 55, 15)[:, :, :, 6:12]
    motion_6d = motion_6d.reshape(bs, -1, 55 * 6)
    return motion_6d


class CustomTrainer(BaseTrainer):
    """
    Generative Trainer to support various generative models.

    Supports two formats:
    - SMPL-X (default): 330D body with 3 VQ models (upper/hands/lower)
    - BVH (use_bvh_format=True): 225D body + 51D face with 2 VQ models
    """

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args

        # Check if using BVH format
        self.use_bvh_format = getattr(cfg, 'use_bvh_format', False)

        if self.use_bvh_format:
            # BVH format: 75 joints, no body part splitting
            self.joints = 75
            self.body_dim = 225  # 75 joints × 3 axis-angle
            self.face_dim = 51   # ARKit blendshapes
            logger.info("Using BVH format: 225D body (axis-angle) + 51D face (ARKit)")
        else:
            # SMPL-X format: 55 joints with body part splitting
            self.joints = 55
            self.ori_joint_list = joints_list["beat_smplx_joints"]
            self.tar_joint_list_face = joints_list["beat_smplx_face"]
            self.tar_joint_list_upper = joints_list["beat_smplx_upper"]
            self.tar_joint_list_hands = joints_list["beat_smplx_hands"]
            self.tar_joint_list_lower = joints_list["beat_smplx_lower"]

            self.joint_mask_face = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
            for joint_name in self.tar_joint_list_face:
                self.joint_mask_face[
                    self.ori_joint_list[joint_name][1]
                    - self.ori_joint_list[joint_name][0] : self.ori_joint_list[joint_name][
                        1
                    ]
                ] = 1
            self.joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
            for joint_name in self.tar_joint_list_upper:
                self.joint_mask_upper[
                    self.ori_joint_list[joint_name][1]
                    - self.ori_joint_list[joint_name][0] : self.ori_joint_list[joint_name][
                        1
                    ]
                ] = 1
            self.joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
            for joint_name in self.tar_joint_list_hands:
                self.joint_mask_hands[
                    self.ori_joint_list[joint_name][1]
                    - self.ori_joint_list[joint_name][0] : self.ori_joint_list[joint_name][
                        1
                    ]
                ] = 1
            self.joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
            for joint_name in self.tar_joint_list_lower:
                self.joint_mask_lower[
                    self.ori_joint_list[joint_name][1]
                    - self.ori_joint_list[joint_name][0] : self.ori_joint_list[joint_name][
                        1
                    ]
                ] = 1

        self.tracker = other_tools.EpochTracker(
            ["fgd", "bc", "l1div", "predict_x0_loss", "test_clip_fgd"],
            [True, True, True, True, True],
        )

        ##### Model #####

        model_module = __import__(
            f"models.{cfg.model.model_name}", fromlist=["something"]
        )

        if self.cfg.ddp:
            self.model = getattr(model_module, cfg.model.g_name)(cfg).to(self.rank)
            process_group = torch.distributed.new_group()
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model, process_group
            )
            self.model = DDP(
                self.model,
                device_ids=[self.rank],
                output_device=self.rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            self.model = torch.nn.DataParallel(
                getattr(model_module, cfg.model.g_name)(cfg), self.cfg.gpus
            ).cuda()

        if self.args.mode == "train":
            if self.rank == 0:
                logger.info(self.model)
                logger.info(f"init {self.cfg.model.g_name} success")
                wandb.watch(self.model)

        ##### Optimizer and Scheduler #####
        self.opt = create_optimizer(self.cfg.solver, self.model)
        self.opt_s = create_scheduler(self.cfg.solver, self.opt)

        ##### VQ-VAE models #####
        """Initialize and load VQ-VAE models for different body parts."""
        if self.use_bvh_format:
            # BVH format: 2 VQ models (body + face)
            self.vq_models = self._create_bvh_vq_models()
            for model in self.vq_models.values():
                model.eval().to(self.rank)
            self.vq_model_body = self.vq_models["body"]
            self.vq_model_face = self.vq_models["face"]
        else:
            # SMPL-X format: 3 VQ models (upper/hands/lower)
            self.vq_models = self._create_body_vq_models()
            for model in self.vq_models.values():
                model.eval().to(self.rank)
            self.vq_model_upper, self.vq_model_hands, self.vq_model_lower = (
                self.vq_models.values()
            )

        ##### Loss functions #####
        self.reclatent_loss = nn.MSELoss().to(self.rank)
        self.vel_loss = torch.nn.L1Loss(reduction="mean").to(self.rank)

        ##### Normalization #####
        if self.use_bvh_format:
            # BVH format: load from VQ-VAE output directory
            body_mean_path = getattr(cfg, 'mean_pose_path', './outputs/vqvae_bvh/body/body_mean.npy')
            body_std_path = getattr(cfg, 'std_pose_path', './outputs/vqvae_bvh/body/body_std.npy')
            face_mean_path = getattr(cfg, 'mean_face_path', './outputs/vqvae_bvh/face/face_mean.npy')
            face_std_path = getattr(cfg, 'std_face_path', './outputs/vqvae_bvh/face/face_std.npy')

            self.mean_body = torch.from_numpy(np.load(body_mean_path)).float().cuda()
            self.std_body = torch.from_numpy(np.load(body_std_path)).float().cuda()
            self.mean_face = torch.from_numpy(np.load(face_mean_path)).float().cuda()
            self.std_face = torch.from_numpy(np.load(face_std_path)).float().cuda()
            logger.info(f"Loaded BVH normalization: body={body_mean_path}, face={face_mean_path}")
        else:
            # SMPL-X format: original paths
            self.mean = np.load("./mean_std/beatx_2_330_mean.npy")
            self.std = np.load("./mean_std/beatx_2_330_std.npy")

            # Extract body part specific normalizations
            for part in ["upper", "hands", "lower"]:
                mask = globals()[f"{part}_body_mask"]
                setattr(self, f"mean_{part}", torch.from_numpy(self.mean[mask]).cuda())
                setattr(self, f"std_{part}", torch.from_numpy(self.std[mask]).cuda())

            self.trans_mean = torch.from_numpy(
                np.load("./mean_std/beatx_2_trans_mean.npy")
            ).cuda()
            self.trans_std = torch.from_numpy(
                np.load("./mean_std/beatx_2_trans_std.npy")
            ).cuda()

        if self.args.checkpoint:
            try:
                ckpt_state_dict = torch.load(self.args.checkpoint, weights_only=False)[
                    "model_state_dict"
                ]
            except:
                ckpt_state_dict = torch.load(self.args.checkpoint, weights_only=False)[
                    "model_state"
                ]
            # remove 'audioEncoder' from the state_dict due to legacy issues
            ckpt_state_dict = {
                k: v
                for k, v in ckpt_state_dict.items()
                if "modality_encoder.audio_encoder." not in k
            }
            self.model.load_state_dict(ckpt_state_dict, strict=False)
            logger.info(f"Loaded checkpoint from {self.args.checkpoint}")

    def _create_body_vq_models(self) -> Dict[str, RVQVAE]:
        """Create VQ-VAE models for SMPL-X body parts."""
        vq_configs = {
            "upper": {"dim_pose": 78},
            "hands": {"dim_pose": 180},
            "lower": {"dim_pose": 57},
        }

        vq_models = {}
        for part, config in vq_configs.items():
            model = self._create_rvqvae_model(config["dim_pose"], part)
            vq_models[part] = model

        return vq_models

    def _create_bvh_vq_models(self) -> Dict[str, RVQVAE]:
        """Create VQ-VAE models for BVH format (body + face)."""
        vq_configs = {
            "body": {"dim_pose": 225},  # 75 joints × 3 axis-angle
            "face": {"dim_pose": 51},   # ARKit blendshapes
        }

        vq_models = {}
        for part, config in vq_configs.items():
            model = self._create_rvqvae_model(config["dim_pose"], part)
            vq_models[part] = model

        return vq_models

    def _create_rvqvae_model(self, dim_pose: int, body_part: str) -> RVQVAE:
        """Create a single RVQVAE model with specified configuration."""

        vq_args = self.args
        vq_args.num_quantizers = 6
        vq_args.shared_codebook = False
        vq_args.quantize_dropout_prob = 0.2
        vq_args.quantize_dropout_cutoff_index = 0
        vq_args.mu = 0.99
        vq_args.beta = 1.0
        model = RVQVAE(
            vq_args,
            input_width=dim_pose,
            nb_code=1024,
            code_dim=128,
            output_emb_width=128,
            down_t=2,
            stride_t=2,
            width=512,
            depth=3,
            dilation_growth_rate=3,
            activation="relu",
            norm=None,
        )

        # Load pretrained weights
        checkpoint_path = getattr(self.cfg, f"vqvae_{body_part}_path")
        model.load_state_dict(torch.load(checkpoint_path)["net"])
        return model

    def inverse_selection(self, filtered_t, selection_array, n):
        original_shape_t = np.zeros((n, selection_array.size))
        selected_indices = np.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t

    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).cuda()
        original_shape_t = torch.zeros((n, 165)).cuda()
        selected_indices = torch.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t

    def _load_data(self, dict_data):
        if self.use_bvh_format:
            return self._load_data_bvh(dict_data)
        else:
            return self._load_data_smplx(dict_data)

    def _load_data_bvh(self, dict_data):
        """Load data for BVH format (225D body + 51D face)."""
        # Motion is already normalized by dataloader: (B, T, 276)
        motion = dict_data["motion"].to(self.rank)
        mel = dict_data["mel"].to(self.rank)

        bs, n, _ = motion.shape

        # Split body and face
        tar_body = motion[:, :, :225]   # (B, T, 225) axis-angle
        tar_face = motion[:, :, 225:]   # (B, T, 51) ARKit

        # Encode to VQ latents
        latent_body = self.vq_model_body.map2latent(tar_body)
        latent_face = self.vq_model_face.map2latent(tar_face)

        # Concatenate latents and scale
        latent_in = torch.cat([latent_body, latent_face], dim=2) / self.cfg.vqvae_latent_scale

        # Get speaker ID if available
        tar_id = dict_data.get("id", torch.zeros(bs, n, dtype=torch.long)).to(self.rank)

        return {
            "audio_onset": mel,  # Use mel as audio input for BVH
            "word": None,
            "latent_in": latent_in,
            "tar_id": tar_id,
            "tar_body": tar_body,
            "tar_face": tar_face,
            "style_feature": None,
        }

    def _load_data_smplx(self, dict_data):
        """Load data for SMPL-X format (original implementation)."""
        facial_rep = dict_data["facial"].to(self.rank)
        beta = dict_data["beta"].to(self.rank)
        tar_trans = dict_data["trans"].to(self.rank)
        tar_id = dict_data["id"].to(self.rank)

        # process the pose data
        tar_pose = dict_data["pose"][:, :, :165].to(self.rank)
        tar_trans_v = dict_data["trans_v"].to(self.rank)
        tar_trans = dict_data["trans"].to(self.rank)
        bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
        tar_pose_hands = tar_pose[:, :, 25 * 3 : 55 * 3]
        tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30 * 6)

        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13 * 6)

        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9 * 6)

        tar_pose_lower = tar_pose_leg

        tar_pose_upper = (tar_pose_upper - self.mean_upper) / self.std_upper
        tar_pose_hands = (tar_pose_hands - self.mean_hands) / self.std_hands
        tar_pose_lower = (tar_pose_lower - self.mean_lower) / self.std_lower

        tar_trans_v = (tar_trans_v - self.trans_mean) / self.trans_std
        tar_pose_lower = torch.cat([tar_pose_lower, tar_trans_v], dim=-1)

        latent_upper_top = self.vq_model_upper.map2latent(tar_pose_upper)
        latent_hands_top = self.vq_model_hands.map2latent(tar_pose_hands)
        latent_lower_top = self.vq_model_lower.map2latent(tar_pose_lower)


        ## TODO: Whether the latent scale is needed here?
        # latent_in = torch.cat([latent_upper_top, latent_hands_top, latent_lower_top], dim=2)
        latent_in = (
            torch.cat([latent_upper_top, latent_hands_top, latent_lower_top], dim=2) / 5
        )

        word = dict_data.get("word", None)
        if word is not None:
            word = word.to(self.rank)

        # style feature is always None (without annotation, we never know what it is)
        style_feature = None

        audio_onset = None
        if self.cfg.data.onset_rep:
            audio_onset = dict_data["audio_onset"].to(self.rank)

        return {
            "audio_onset": audio_onset,
            "word": word,
            "latent_in": latent_in,
            "tar_id": tar_id,
            "facial_rep": facial_rep,
            "beta": beta,
            "tar_pose": tar_pose,
            "trans": tar_trans,
            "style_feature": style_feature,
        }

    def _g_training(self, loaded_data, mode="train", epoch=0):
        self.model.train()
        cond_ = {"y": {}}
        cond_["y"]["audio_onset"] = loaded_data["audio_onset"]
        cond_["y"]["word"] = loaded_data["word"]
        cond_["y"]["id"] = loaded_data["tar_id"]
        cond_["y"]["seed"] = loaded_data["latent_in"][:, : self.cfg.pre_frames]
        cond_["y"]["style_feature"] = loaded_data["style_feature"]
        x0 = loaded_data["latent_in"]
        x0 = x0.permute(0, 2, 1).unsqueeze(2)

        g_loss_final = self.model.module.train_forward(cond_, x0)["loss"]

        self.tracker.update_meter("predict_x0_loss", "train", g_loss_final.item())

        if mode == "train":
            return g_loss_final

    def _g_test(self, loaded_data):
        if self.use_bvh_format:
            return self._g_test_bvh(loaded_data)
        else:
            return self._g_test_smplx(loaded_data)

    def _g_test_bvh(self, loaded_data):
        """Test/inference for BVH format."""
        self.model.eval()

        tar_body = loaded_data["tar_body"]
        tar_face = loaded_data["tar_face"]
        audio_onset = loaded_data["audio_onset"]  # mel spectrogram
        in_seed = loaded_data["latent_in"]

        bs, n, _ = tar_body.shape
        code_dim = self.vq_model_body.code_dim

        # Handle frame alignment
        remain = n % 8
        if remain != 0:
            tar_body = tar_body[:, :-remain, :]
            tar_face = tar_face[:, :-remain, :]
            n = n - remain

        vqvae_squeeze_scale = self.cfg.vqvae_squeeze_scale
        pre_frames_scaled = self.cfg.pre_frames * vqvae_squeeze_scale

        # Generate latents
        rec_all_body = []
        rec_all_face = []

        roundt = max(1, (n - pre_frames_scaled) // (self.cfg.pose_length - pre_frames_scaled))

        for i in range(roundt):
            round_l = self.cfg.pose_length - pre_frames_scaled

            # Get audio chunk
            if audio_onset is not None:
                mel_start = i * round_l
                mel_end = (i + 1) * round_l + pre_frames_scaled
                in_audio_tmp = audio_onset[:, mel_start:mel_end, :]
            else:
                in_audio_tmp = None

            # Get seed
            if i == 0:
                in_seed_tmp = in_seed[:, :self.cfg.pre_frames, :]
            else:
                in_seed_tmp = last_sample[:, -self.cfg.pre_frames:, :]

            # Prepare conditioning
            cond_ = {"y": {}}
            cond_["y"]["audio_onset"] = in_audio_tmp
            cond_["y"]["word"] = None
            cond_["y"]["id"] = loaded_data["tar_id"][:, :round_l + pre_frames_scaled] if "tar_id" in loaded_data else None
            cond_["y"]["seed"] = in_seed_tmp
            cond_["y"]["style_feature"] = None

            # Generate
            sample = self.model(cond_)["latents"]
            sample = sample.squeeze(2).permute(0, 2, 1)
            last_sample = sample.clone()

            # Split body and face latents
            rec_latent_body = sample[..., :code_dim]
            rec_latent_face = sample[..., code_dim:code_dim * 2]

            if i == 0:
                rec_all_body.append(rec_latent_body)
                rec_all_face.append(rec_latent_face)
            else:
                rec_all_body.append(rec_latent_body[:, self.cfg.pre_frames:])
                rec_all_face.append(rec_latent_face[:, self.cfg.pre_frames:])

        # Concatenate and decode
        rec_all_body = torch.cat(rec_all_body, dim=1) * self.cfg.vqvae_latent_scale
        rec_all_face = torch.cat(rec_all_face, dim=1) * self.cfg.vqvae_latent_scale

        # latent2origin returns (B, dim, T), permute to (B, T, dim)
        rec_body = self.vq_model_body.latent2origin(rec_all_body)[0].permute(0, 2, 1)
        rec_face = self.vq_model_face.latent2origin(rec_all_face)[0].permute(0, 2, 1)

        # Denormalize
        rec_body = rec_body * self.std_body + self.mean_body
        rec_face = rec_face * self.std_face + self.mean_face

        # Align lengths
        min_len = min(rec_body.shape[1], tar_body.shape[1])
        rec_body = rec_body[:, :min_len]
        rec_face = rec_face[:, :min_len]
        tar_body = tar_body[:, :min_len]
        tar_face = tar_face[:, :min_len]

        return {
            "rec_pose": rec_body,
            "rec_face": rec_face,
            "tar_pose": tar_body,
            "tar_face": tar_face,
        }

    def _g_test_smplx(self, loaded_data):
        """Test/inference for SMPL-X format (original implementation)."""
        self.model.eval()
        tar_beta = loaded_data["beta"]
        tar_pose = loaded_data["tar_pose"]
        tar_exps = loaded_data["facial_rep"]
        tar_trans = loaded_data["trans"]

        audio_onset = loaded_data["audio_onset"]
        in_word = loaded_data["word"]

        in_x0 = loaded_data["latent_in"]
        in_seed = loaded_data["latent_in"]

        bs, n, j = (
            loaded_data["tar_pose"].shape[0],
            loaded_data["tar_pose"].shape[1],
            self.joints,
        )

        remain = n % 8
        if remain != 0:

            tar_pose = tar_pose[:, :-remain, :]
            tar_beta = tar_beta[:, :-remain, :]
            tar_exps = tar_exps[:, :-remain, :]
            in_x0 = in_x0[
                :, : in_x0.shape[1] - (remain // self.cfg.vqvae_squeeze_scale), :
            ]
            in_seed = in_seed[
                :, : in_x0.shape[1] - (remain // self.cfg.vqvae_squeeze_scale), :
            ]
            in_word = in_word[:, :-remain]
            n = n - remain

        rec_all_upper = []
        rec_all_lower = []
        rec_all_hands = []
        vqvae_squeeze_scale = self.cfg.vqvae_squeeze_scale
        pre_frames_scaled = self.cfg.pre_frames * vqvae_squeeze_scale
        roundt = (n - pre_frames_scaled) // (
            self.cfg.data.pose_length - pre_frames_scaled
        )
        remain = (n - pre_frames_scaled) % (
            self.cfg.data.pose_length - pre_frames_scaled
        )
        round_l = self.cfg.pose_length - pre_frames_scaled
        round_audio = int(round_l / 3 * 5)

        in_audio_onset_tmp = None
        in_word_tmp = None
        for i in range(0, roundt):
            if audio_onset is not None:
                in_audio_onset_tmp = audio_onset[
                    :,
                    i * (16000 // 30 * round_l) : (i + 1) * (16000 // 30 * round_l)
                    + 16000 // 30 * self.cfg.pre_frames * vqvae_squeeze_scale,
                ]
            if in_word is not None:
                in_word_tmp = in_word[
                    :,
                    i * (round_l) : (i + 1) * (round_l)
                    + self.cfg.pre_frames * vqvae_squeeze_scale,
                ]

            in_id_tmp = loaded_data["tar_id"][
                :, i * (round_l) : (i + 1) * (round_l) + self.cfg.pre_frames
            ]
            in_seed_tmp = in_seed[
                :,
                i
                * (round_l)
                // vqvae_squeeze_scale : (i + 1)
                * (round_l)
                // vqvae_squeeze_scale
                + self.cfg.pre_frames,
            ]

            if i == 0:
                in_seed_tmp = in_seed_tmp[:, : self.cfg.pre_frames, :]
            else:
                in_seed_tmp = last_sample[:, -self.cfg.pre_frames :, :]

            cond_ = {"y": {}}
            cond_["y"]["audio_onset"] = in_audio_onset_tmp
            cond_["y"]["word"] = in_word_tmp
            cond_["y"]["id"] = in_id_tmp
            cond_["y"]["seed"] = in_seed_tmp
            cond_["y"]["style_feature"] = torch.zeros([bs, 512]).cuda()

            sample = self.model(cond_)["latents"]

            sample = sample.squeeze(2).permute(0, 2, 1)

            last_sample = sample.clone()

            code_dim = self.vq_model_upper.code_dim
            rec_latent_upper = sample[..., :code_dim]
            rec_latent_hands = sample[..., code_dim : code_dim * 2]
            rec_latent_lower = sample[..., code_dim * 2 : code_dim * 3]

            if i == 0:
                rec_all_upper.append(rec_latent_upper)
                rec_all_hands.append(rec_latent_hands)
                rec_all_lower.append(rec_latent_lower)
            else:
                rec_all_upper.append(rec_latent_upper[:, self.cfg.pre_frames :])
                rec_all_hands.append(rec_latent_hands[:, self.cfg.pre_frames :])
                rec_all_lower.append(rec_latent_lower[:, self.cfg.pre_frames :])

        rec_all_upper = torch.cat(rec_all_upper, dim=1) * 5
        rec_all_hands = torch.cat(rec_all_hands, dim=1) * 5
        rec_all_lower = torch.cat(rec_all_lower, dim=1) * 5

        rec_upper = self.vq_model_upper.latent2origin(rec_all_upper)[0]
        rec_hands = self.vq_model_hands.latent2origin(rec_all_hands)[0]
        rec_lower = self.vq_model_lower.latent2origin(rec_all_lower)[0]

        rec_trans_v = rec_lower[..., -3:]
        rec_trans_v = rec_trans_v * self.trans_std + self.trans_mean
        rec_trans = torch.zeros_like(rec_trans_v)
        rec_trans = torch.cumsum(rec_trans_v, dim=-2)
        rec_trans[..., 1] = rec_trans_v[..., 1]
        rec_lower = rec_lower[..., :-3]

        rec_upper = rec_upper * self.std_upper + self.mean_upper
        rec_hands = rec_hands * self.std_hands + self.mean_hands
        rec_lower = rec_lower * self.std_lower + self.mean_lower

        n = n - remain
        tar_pose = tar_pose[:, :n, :]
        tar_exps = tar_exps[:, :n, :]
        tar_trans = tar_trans[:, :n, :]
        tar_beta = tar_beta[:, :n, :]

        if hasattr(self.cfg.model, "use_exp") and self.cfg.model.use_exp:
            rec_exps = tar_exps  # fallback to tar_exps since rec_face is not defined
        else:
            rec_exps = tar_exps

        rec_trans = tar_trans

        rec_pose_legs = rec_lower[:, :, :54]
        bs, n = rec_pose_legs.shape[0], rec_pose_legs.shape[1]
        rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
        rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)  #
        rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs * n, 13 * 3)
        rec_pose_upper_recover = self.inverse_selection_tensor(
            rec_pose_upper, self.joint_mask_upper, bs * n
        )
        rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
        rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)

        rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs * n, 9 * 3)
        rec_pose_lower_recover = self.inverse_selection_tensor(
            rec_pose_lower, self.joint_mask_lower, bs * n
        )
        rec_pose_hands = rec_hands.reshape(bs, n, 30, 6)
        rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs * n, 30 * 3)
        rec_pose_hands_recover = self.inverse_selection_tensor(
            rec_pose_hands, self.joint_mask_hands, bs * n
        )
        rec_pose = (
            rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
        )
        rec_pose[:, 66:69] = tar_pose.reshape(bs * n, 55 * 3)[:, 66:69]

        rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs * n, j, 3))
        rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j * 6)
        tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs * n, j, 3))
        tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j * 6)

        return {
            "rec_pose": rec_pose,
            "rec_exps": rec_exps,
            "rec_trans": rec_trans,
            "tar_pose": tar_pose,
            "tar_exps": tar_exps,
            "tar_beta": tar_beta,
            "tar_trans": tar_trans,
        }

    def train(self, epoch):

        self.model.train()
        t_start = time.time()
        self.tracker.reset()
        for its, batch_data in enumerate(self.train_loader):
            loaded_data = self._load_data(batch_data)
            t_data = time.time() - t_start

            self.opt.zero_grad()
            g_loss_final = 0
            g_loss_final += self._g_training(loaded_data, "train", epoch)

            g_loss_final.backward()
            if self.cfg.solver.grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.solver.grad_norm
                )
            self.opt.step()

            mem_cost = torch.cuda.memory_cached() / 1e9
            lr_g = self.opt.param_groups[0]["lr"]

            t_train = time.time() - t_start - t_data
            t_start = time.time()
            if its % self.cfg.log_period == 0:
                self.train_recording(epoch, its, t_data, t_train, mem_cost, lr_g)
            if self.cfg.debug:
                if its == 1:
                    break
        self.opt_s.step(epoch)

    @torch.no_grad()
    def _common_test_inference(
        self, data_loader, epoch, mode="val", max_iterations=None, save_results=False
    ):
        """
        Common inference logic shared by val, test, test_clip, and test_render methods.

        Args:
            data_loader: The data loader to iterate over
            epoch: Current epoch number
            mode: Mode string for logging ("val", "test", "test_clip", "test_render")
            max_iterations: Maximum number of iterations (None for no limit)
            save_results: Whether to save result files

        Returns:
            Dictionary containing computed metrics and results
        """
        start_time = time.time()
        total_length = 0
        test_seq_list = self.test_data.selected_file
        align = 0
        latent_out = []
        latent_ori = []
        l2_all = 0
        lvel = 0
        results = []

        # Setup save path for test mode
        results_save_path = None
        if save_results:
            results_save_path = self.checkpoint_path + f"/{epoch}/"
            if mode == "test_render":
                if os.path.exists(results_save_path):
                    import shutil

                    shutil.rmtree(results_save_path)
            os.makedirs(results_save_path, exist_ok=True)

        self.model.eval()
        self.smplx.eval()
        if hasattr(self, "eval_copy"):
            self.eval_copy.eval()

        with torch.no_grad():
            iterator = enumerate(data_loader)
            if mode in ["test_clip", "test"]:
                iterator = enumerate(
                    tqdm(data_loader, desc=f"Testing {mode}", leave=True)
                )

            for its, batch_data in iterator:
                if max_iterations is not None and its > max_iterations:
                    break

                loaded_data = self._load_data(batch_data)
                net_out = self._g_test(loaded_data)

                tar_pose = net_out["tar_pose"]
                rec_pose = net_out["rec_pose"]
                tar_exps = net_out["tar_exps"]
                tar_beta = net_out["tar_beta"]
                rec_trans = net_out["rec_trans"]
                tar_trans = net_out.get("tar_trans", rec_trans)
                rec_exps = net_out.get("rec_exps", tar_exps)

                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints

                # Handle frame rate conversion
                if (30 / self.cfg.data.pose_fps) != 1:
                    assert 30 % self.cfg.data.pose_fps == 0
                    n *= int(30 / self.cfg.data.pose_fps)
                    tar_pose = torch.nn.functional.interpolate(
                        tar_pose.permute(0, 2, 1),
                        scale_factor=30 / self.cfg.data.pose_fps,
                        mode="linear",
                    ).permute(0, 2, 1)
                    scale_factor = (
                        30 / self.cfg.data.pose_fps
                        if mode != "test"
                        else 30 / self.cfg.pose_fps
                    )
                    rec_pose = torch.nn.functional.interpolate(
                        rec_pose.permute(0, 2, 1),
                        scale_factor=scale_factor,
                        mode="linear",
                    ).permute(0, 2, 1)

                # Calculate latent representations for evaluation
                if hasattr(self, "eval_copy") and mode != "test_render":
                    remain = n % self.cfg.vae_test_len
                    latent_out.append(
                        self.eval_copy.map2latent(rec_pose[:, : n - remain])
                        .reshape(-1, self.cfg.vae_length)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    latent_ori.append(
                        self.eval_copy.map2latent(tar_pose[:, : n - remain])
                        .reshape(-1, self.cfg.vae_length)
                        .detach()
                        .cpu()
                        .numpy()
                    )

                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs * n, j, 6))
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs * n, j * 3)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs * n, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs * n, j * 3)

                # Generate SMPLX vertices and joints
                vertices_rec = self.smplx(
                    betas=tar_beta.reshape(bs * n, 300),
                    transl=rec_trans.reshape(bs * n, 3) - rec_trans.reshape(bs * n, 3),
                    expression=tar_exps.reshape(bs * n, 100)
                    - tar_exps.reshape(bs * n, 100),
                    jaw_pose=rec_pose[:, 66:69],
                    global_orient=rec_pose[:, :3],
                    body_pose=rec_pose[:, 3 : 21 * 3 + 3],
                    left_hand_pose=rec_pose[:, 25 * 3 : 40 * 3],
                    right_hand_pose=rec_pose[:, 40 * 3 : 55 * 3],
                    return_joints=True,
                    leye_pose=rec_pose[:, 69:72],
                    reye_pose=rec_pose[:, 72:75],
                )

                joints_rec = (
                    vertices_rec["joints"]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(bs, n, 127 * 3)[0, :n, : 55 * 3]
                )

                # Calculate L1 diversity
                if hasattr(self, "l1_calculator"):
                    _ = self.l1_calculator.run(joints_rec)

                # Calculate alignment for single batch
                if (
                    hasattr(self, "alignmenter")
                    and self.alignmenter is not None
                    and bs == 1
                    and mode != "test_render"
                ):
                    in_audio_eval, sr = librosa.load(
                        self.cfg.data.data_path
                        + "wave16k/"
                        + test_seq_list.iloc[its]["id"]
                        + ".wav"
                    )
                    in_audio_eval = librosa.resample(
                        in_audio_eval, orig_sr=sr, target_sr=self.cfg.data.audio_sr
                    )
                    a_offset = int(
                        self.align_mask
                        * (self.cfg.data.audio_sr / self.cfg.data.pose_fps)
                    )
                    onset_bt = self.alignmenter.load_audio(
                        in_audio_eval[
                            : int(self.cfg.data.audio_sr / self.cfg.data.pose_fps * n)
                        ],
                        a_offset,
                        len(in_audio_eval) - a_offset,
                        True,
                    )
                    beat_vel = self.alignmenter.load_pose(
                        joints_rec, self.align_mask, n - self.align_mask, 30, True
                    )
                    align += self.alignmenter.calculate_align(
                        onset_bt, beat_vel, 30
                    ) * (n - 2 * self.align_mask)

                # Mode-specific processing
                if mode == "test" and save_results:
                    # Calculate facial losses for test mode
                    vertices_rec_face = self.smplx(
                        betas=tar_beta.reshape(bs * n, 300),
                        transl=rec_trans.reshape(bs * n, 3)
                        - rec_trans.reshape(bs * n, 3),
                        expression=rec_exps.reshape(bs * n, 100),
                        jaw_pose=rec_pose[:, 66:69],
                        global_orient=rec_pose[:, :3] - rec_pose[:, :3],
                        body_pose=rec_pose[:, 3 : 21 * 3 + 3]
                        - rec_pose[:, 3 : 21 * 3 + 3],
                        left_hand_pose=rec_pose[:, 25 * 3 : 40 * 3]
                        - rec_pose[:, 25 * 3 : 40 * 3],
                        right_hand_pose=rec_pose[:, 40 * 3 : 55 * 3]
                        - rec_pose[:, 40 * 3 : 55 * 3],
                        return_verts=True,
                        return_joints=True,
                        leye_pose=rec_pose[:, 69:72] - rec_pose[:, 69:72],
                        reye_pose=rec_pose[:, 72:75] - rec_pose[:, 72:75],
                    )
                    vertices_tar_face = self.smplx(
                        betas=tar_beta.reshape(bs * n, 300),
                        transl=tar_trans.reshape(bs * n, 3)
                        - tar_trans.reshape(bs * n, 3),
                        expression=tar_exps.reshape(bs * n, 100),
                        jaw_pose=tar_pose[:, 66:69],
                        global_orient=tar_pose[:, :3] - tar_pose[:, :3],
                        body_pose=tar_pose[:, 3 : 21 * 3 + 3]
                        - tar_pose[:, 3 : 21 * 3 + 3],
                        left_hand_pose=tar_pose[:, 25 * 3 : 40 * 3]
                        - tar_pose[:, 25 * 3 : 40 * 3],
                        right_hand_pose=tar_pose[:, 40 * 3 : 55 * 3]
                        - tar_pose[:, 40 * 3 : 55 * 3],
                        return_verts=True,
                        return_joints=True,
                        leye_pose=tar_pose[:, 69:72] - tar_pose[:, 69:72],
                        reye_pose=tar_pose[:, 72:75] - tar_pose[:, 72:75],
                    )

                    facial_rec = (
                        vertices_rec_face["vertices"].reshape(1, n, -1)[0, :n].cpu()
                    )
                    facial_tar = (
                        vertices_tar_face["vertices"].reshape(1, n, -1)[0, :n].cpu()
                    )
                    face_vel_loss = self.vel_loss(
                        facial_rec[1:, :] - facial_tar[:-1, :],
                        facial_tar[1:, :] - facial_tar[:-1, :],
                    )
                    l2 = self.reclatent_loss(facial_rec, facial_tar)
                    l2_all += l2.item() * n
                    lvel += face_vel_loss.item() * n

                # Save results if needed
                if save_results:
                    if mode == "test":
                        # Save NPZ files for test mode
                        tar_pose_np = tar_pose.detach().cpu().numpy()
                        rec_pose_np = rec_pose.detach().cpu().numpy()
                        rec_trans_np = (
                            rec_trans.detach().cpu().numpy().reshape(bs * n, 3)
                        )
                        rec_exp_np = (
                            rec_exps.detach().cpu().numpy().reshape(bs * n, 100)
                        )
                        tar_exp_np = (
                            tar_exps.detach().cpu().numpy().reshape(bs * n, 100)
                        )
                        tar_trans_np = (
                            tar_trans.detach().cpu().numpy().reshape(bs * n, 3)
                        )

                        gt_npz = np.load(
                            self.cfg.data.data_path
                            + self.cfg.data.pose_rep
                            + "/"
                            + test_seq_list.iloc[its]["id"]
                            + ".npz",
                            allow_pickle=True,
                        )

                        np.savez(
                            results_save_path
                            + "gt_"
                            + test_seq_list.iloc[its]["id"]
                            + ".npz",
                            betas=gt_npz["betas"],
                            poses=tar_pose_np,
                            expressions=tar_exp_np,
                            trans=tar_trans_np,
                            model="smplx2020",
                            gender="neutral",
                            mocap_frame_rate=30,
                        )
                        np.savez(
                            results_save_path
                            + "res_"
                            + test_seq_list.iloc[its]["id"]
                            + ".npz",
                            betas=gt_npz["betas"],
                            poses=rec_pose_np,
                            expressions=rec_exp_np,
                            trans=rec_trans_np,
                            model="smplx2020",
                            gender="neutral",
                            mocap_frame_rate=30,
                        )

                    elif mode == "test_render":
                        # Save results and render for test_render mode
                        audio_name = loaded_data["audio_name"][0]
                        rec_pose_np = rec_pose.detach().cpu().numpy()
                        rec_trans_np = (
                            rec_trans.detach().cpu().numpy().reshape(bs * n, 3)
                        )
                        rec_exp_np = (
                            rec_exps.detach().cpu().numpy().reshape(bs * n, 100)
                        )

                        gt_npz = np.load(
                            "./demo/examples/2_scott_0_1_1.npz", allow_pickle=True
                        )
                        file_name = audio_name.split("/")[-1].split(".")[0]
                        results_npz_file_save_path = (
                            results_save_path + f"result_{file_name}.npz"
                        )

                        np.savez(
                            results_npz_file_save_path,
                            betas=gt_npz["betas"],
                            poses=rec_pose_np,
                            expressions=rec_exp_np,
                            trans=rec_trans_np,
                            model="smplx2020",
                            gender="neutral",
                            mocap_frame_rate=30,
                        )

                        render_vid_path = other_tools_hf.render_one_sequence_no_gt(
                            results_npz_file_save_path,
                            results_save_path,
                            audio_name,
                            self.cfg.data_path_1 + "smplx_models/",
                            use_matplotlib=False,
                            args=self.cfg,
                        )

                total_length += n

        return {
            "total_length": total_length,
            "align": align,
            "latent_out": latent_out,
            "latent_ori": latent_ori,
            "l2_all": l2_all,
            "lvel": lvel,
            "start_time": start_time,
        }

    def val(self, epoch):
        if self.use_bvh_format:
            return self._val_bvh(epoch)
        else:
            return self._val_smplx(epoch)

    def _val_bvh(self, epoch):
        """Simplified validation for BVH format (no SMPL-X metrics)."""
        self.tracker.reset()
        self.model.eval()

        total_loss = 0
        total_samples = 0
        start_time = time.time()

        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                if its > 15:  # Limit validation iterations
                    break

                loaded_data = self._load_data(batch_data)
                net_out = self._g_test_bvh(loaded_data)

                # Compute simple reconstruction loss
                tar_body = net_out["tar_pose"]
                rec_body = net_out["rec_pose"]
                tar_face = net_out["tar_face"]
                rec_face = net_out["rec_face"]

                # MSE loss for body and face
                body_loss = F.mse_loss(rec_body, tar_body).item()
                face_loss = F.mse_loss(rec_face, tar_face).item()

                total_loss += (body_loss + face_loss) * tar_body.shape[0]
                total_samples += tar_body.shape[0]

        avg_loss = total_loss / max(total_samples, 1)

        # Use loss as FGD proxy for BVH mode (lower is better)
        self.tracker.update_meter("fgd", "val", avg_loss)
        self.tracker.update_meter("bc", "val", 0.0)  # Placeholder
        self.tracker.update_meter("l1div", "val", 0.0)  # Placeholder

        logger.info(f"BVH val loss: {avg_loss:.6f}")
        self.val_recording(epoch)

        end_time = time.time() - start_time
        logger.info(f"BVH validation time: {int(end_time)} s")

    def _val_smplx(self, epoch):
        """Original validation for SMPL-X format."""
        self.tracker.reset()

        results = self._common_test_inference(
            self.test_loader, epoch, mode="val", max_iterations=15
        )

        total_length = results["total_length"]
        align = results["align"]
        latent_out = results["latent_out"]
        latent_ori = results["latent_ori"]
        l2_all = results["l2_all"]
        lvel = results["lvel"]
        start_time = results["start_time"]

        logger.info(f"l2 loss: {l2_all/total_length:.10f}")
        logger.info(f"lvel loss: {lvel/total_length:.10f}")

        latent_out_all = np.concatenate(latent_out, axis=0)
        latent_ori_all = np.concatenate(latent_ori, axis=0)

        fgd = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        logger.info(f"fgd score: {fgd}")
        self.tracker.update_meter("fgd", "val", fgd)

        align_avg = align / (total_length - 2 * len(self.test_loader) * self.align_mask)
        logger.info(f"align score: {align_avg}")
        self.tracker.update_meter("bc", "val", align_avg)

        l1div = self.l1_calculator.avg()
        logger.info(f"l1div score: {l1div}")
        self.tracker.update_meter("l1div", "val", l1div)

        self.val_recording(epoch)

        end_time = time.time() - start_time
        logger.info(
            f"total inference time: {int(end_time)} s for {int(total_length/self.cfg.data.pose_fps)} s motion"
        )

    def test_clip(self, epoch):
        # BVH mode doesn't use test_clip, fall back to regular validation
        if self.use_bvh_format:
            logger.info("BVH mode: test_clip not supported, using regular validation")
            return self._val_bvh(epoch)

        self.tracker.reset()

        # Test on CLIP dataset
        results_clip = self._common_test_inference(
            self.test_clip_loader, epoch, mode="test_clip"
        )

        total_length_clip = results_clip["total_length"]
        latent_out_clip = results_clip["latent_out"]
        latent_ori_clip = results_clip["latent_ori"]
        start_time = results_clip["start_time"]

        latent_out_all_clip = np.concatenate(latent_out_clip, axis=0)
        latent_ori_all_clip = np.concatenate(latent_ori_clip, axis=0)

        fgd_clip = data_tools.FIDCalculator.frechet_distance(
            latent_out_all_clip, latent_ori_all_clip
        )
        logger.info(f"test_clip fgd score: {fgd_clip}")
        self.tracker.update_meter("test_clip_fgd", "val", fgd_clip)

        current_time = time.time()
        test_clip_time = current_time - start_time
        logger.info(
            f"total test_clip inference time: {int(test_clip_time)} s for {int(total_length_clip/self.cfg.data.pose_fps)} s motion"
        )

        # Test on regular test dataset for recording
        results_test = self._common_test_inference(
            self.test_loader, epoch, mode="test_clip"
        )

        total_length = results_test["total_length"]
        align = results_test["align"]
        latent_out = results_test["latent_out"]
        latent_ori = results_test["latent_ori"]

        latent_out_all = np.concatenate(latent_out, axis=0)
        latent_ori_all = np.concatenate(latent_ori, axis=0)

        fgd = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        logger.info(f"fgd score: {fgd}")
        self.tracker.update_meter("fgd", "val", fgd)

        align_avg = align / (total_length - 2 * len(self.test_loader) * self.align_mask)
        logger.info(f"align score: {align_avg}")
        self.tracker.update_meter("bc", "val", align_avg)

        l1div = self.l1_calculator.avg()
        logger.info(f"l1div score: {l1div}")
        self.tracker.update_meter("l1div", "val", l1div)

        self.val_recording(epoch)

        end_time = time.time() - current_time
        logger.info(
            f"total inference time: {int(end_time)} s for {int(total_length/self.cfg.data.pose_fps)} s motion"
        )

    def test(self, epoch):
        # BVH mode uses simplified testing
        if self.use_bvh_format:
            return self._test_bvh(epoch)
        return self._test_smplx(epoch)

    def _test_bvh(self, epoch):
        """Simplified testing for BVH format - saves generated motion."""
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        os.makedirs(results_save_path, exist_ok=True)

        self.model.eval()
        total_samples = 0
        start_time = time.time()

        with torch.no_grad():
            for its, batch_data in enumerate(tqdm(self.test_loader, desc="Testing BVH")):
                loaded_data = self._load_data(batch_data)
                net_out = self._g_test_bvh(loaded_data)

                rec_body = net_out["rec_pose"].cpu().numpy()
                rec_face = net_out["rec_face"].cpu().numpy()
                tar_body = net_out["tar_pose"].cpu().numpy()
                tar_face = net_out["tar_face"].cpu().numpy()

                # Save results
                np.savez(
                    os.path.join(results_save_path, f"sample_{its}.npz"),
                    rec_body=rec_body,
                    rec_face=rec_face,
                    tar_body=tar_body,
                    tar_face=tar_face,
                )

                total_samples += rec_body.shape[0]

        end_time = time.time() - start_time
        logger.info(f"BVH test: saved {total_samples} samples to {results_save_path}")
        logger.info(f"Total test time: {int(end_time)} s")

    def _test_smplx(self, epoch):
        """Original test method for SMPL-X format."""
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        os.makedirs(results_save_path, exist_ok=True)

        results = self._common_test_inference(
            self.test_loader, epoch, mode="test", save_results=True
        )

        total_length = results["total_length"]
        align = results["align"]
        latent_out = results["latent_out"]
        latent_ori = results["latent_ori"]
        l2_all = results["l2_all"]
        lvel = results["lvel"]
        start_time = results["start_time"]

        logger.info(f"l2 loss: {l2_all/total_length:.10f}")
        logger.info(f"lvel loss: {lvel/total_length:.10f}")

        latent_out_all = np.concatenate(latent_out, axis=0)
        latent_ori_all = np.concatenate(latent_ori, axis=0)
        fgd = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        logger.info(f"fgd score: {fgd}")
        self.test_recording("fgd", fgd, epoch)

        align_avg = align / (total_length - 2 * len(self.test_loader) * self.align_mask)
        logger.info(f"align score: {align_avg}")
        self.test_recording("bc", align_avg, epoch)

        l1div = self.l1_calculator.avg()
        logger.info(f"l1div score: {l1div}")
        self.test_recording("l1div", l1div, epoch)

        end_time = time.time() - start_time
        logger.info(
            f"total inference time: {int(end_time)} s for {int(total_length/self.cfg.data.pose_fps)} s motion"
        )

    def test_render(self, epoch):
        # BVH mode doesn't support rendering (no SMPL-X)
        if self.use_bvh_format:
            logger.info("BVH mode: test_render not supported, use test() instead")
            return self._test_bvh(epoch)

        import platform

        if platform.system() == "Linux":
            os.environ["PYOPENGL_PLATFORM"] = "egl"

        """
        input audio and text, output motion
        do not calculate loss and metric
        save video
        """
        results = self._common_test_inference(
            self.test_loader, epoch, mode="test_render", save_results=True
        )

    def load_checkpoint(self, checkpoint):
        # checkpoint is already a dict, do NOT call torch.load again!
        try:
            ckpt_state_dict = checkpoint["model_state_dict"]
        except:
            ckpt_state_dict = checkpoint["model_state"]
        # remove 'audioEncoder' from the state_dict due to legacy issues
        ckpt_state_dict = {
            k: v
            for k, v in ckpt_state_dict.items()
            if "modality_encoder.audio_encoder." not in k
        }
        self.model.load_state_dict(ckpt_state_dict, strict=False)
        try:
            self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
        except:
            print("No optimizer loaded!")
        if (
            "scheduler_state_dict" in checkpoint
            and checkpoint["scheduler_state_dict"] is not None
        ):
            self.opt_s.load_state_dict(checkpoint["scheduler_state_dict"])
        if "val_best" in checkpoint:
            self.val_best = checkpoint["val_best"]
        logger.info("Checkpoint loaded successfully.")
