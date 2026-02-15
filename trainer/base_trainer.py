# from system_utils import get_gpt_id
# dev = get_gpt_id()
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import signal
import time
import csv
import sys
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import numpy as np
import time
import pprint
from loguru import logger
import smplx
from torch.utils.tensorboard import SummaryWriter
import wandb
import matplotlib.pyplot as plt
from utils import logger_tools, other_tools, metric
import shutil
import argparse
from omegaconf import OmegaConf
from datetime import datetime
import importlib
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_collate
from dataloaders.build_vocab import Vocab


class BaseTrainer(object):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.rank = 0
        self.checkpoint_path = os.path.join(cfg.output_dir, cfg.exp_name)

        # Check if using BVH format (simpler pipeline without SMPL-X)
        self.use_bvh_format = getattr(cfg, 'use_bvh_format', False)

        # Initialize best metrics tracking
        self.val_best = {
            "fgd": {"value": float('inf'), "epoch": 0},  # Add fgd if not present
            "l1div": {"value": float('-inf'), "epoch": 0},  # Higher is better, so start with -inf
            "bc": {"value": float('-inf'), "epoch": 0},  # Higher is better, so start with -inf
            "test_clip_fgd": {"value": float('inf'), "epoch": 0},
        }

        # Initialize data loaders based on format
        if self.use_bvh_format:
            self._init_bvh_dataloaders(cfg)
        else:
            self._init_smplx_dataloaders(cfg)

        self.train_length = len(self.train_loader)
        logger.info(f"Init train and test dataloader successfully")


        if args.mode == "train":
            # Setup logging with wandb
            if self.rank == 0:
                run_time = datetime.now().strftime("%Y%m%d-%H%M")
                run_name = cfg.exp_name + "_" + run_time
                if hasattr(cfg, 'resume_from_checkpoint') and cfg.resume_from_checkpoint:
                    run_name += f"_resumed"

                wandb.init(
                    project=cfg.wandb_project,
                    name=run_name,
                    entity=getattr(cfg, 'wandb_entity', None),
                    dir=getattr(cfg, 'wandb_log_dir', './outputs/wandb/'),
                    config=OmegaConf.to_container(cfg)
                )

        # Skip SMPL-X specific initialization for BVH format
        if self.use_bvh_format:
            logger.info("BVH mode: Skipping SMPL-X and eval model initialization")
            self.eval_copy = None
            self.smplx = None
            self.alignmenter = None
            self.l1_calculator = None
            self.align_mask = 60
        else:
            self._init_smplx_evaluation(cfg)

    def _init_bvh_dataloaders(self, cfg):
        """Initialize dataloaders for BVH format."""
        from dataloaders.beat_normalized import BEATNormalizedDataset

        # Create args object for dataset
        data_args = type('Args', (), {})()
        data_args.data_path = cfg.data.data_path
        data_args.cache_path = cfg.data.cache_path
        data_args.pose_fps = cfg.data.pose_fps
        data_args.audio_sr = cfg.data.audio_sr
        data_args.pose_length = cfg.data.pose_length
        data_args.stride = cfg.data.stride
        data_args.training_speakers = cfg.data.training_speakers
        data_args.new_cache = cfg.data.new_cache

        self.train_data = BEATNormalizedDataset(data_args, split='train')

        if cfg.ddp:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_data)
            self.train_loader = DataLoader(
                self.train_data, batch_size=cfg.data.train_bs,
                sampler=self.train_sampler, drop_last=True, num_workers=4
            )
        else:
            self.train_sampler = None
            self.train_loader = DataLoader(
                self.train_data, batch_size=cfg.data.train_bs,
                shuffle=True, drop_last=True, num_workers=4
            )

        # Test data (same dataset, different split)
        # Skip if skip_eval_model is True to avoid building test cache
        if getattr(cfg, 'skip_eval_model', False):
            logger.info("Skipping test dataloader (skip_eval_model=True)")
            self.test_data = None
            self.test_loader = None
        else:
            self.test_data = BEATNormalizedDataset(data_args, split='test')
            self.test_loader = DataLoader(self.test_data, batch_size=1, drop_last=False)

        # No test_clip for BVH mode
        self.test_clip_data = None
        self.test_clip_loader = None

        if self.test_data:
            logger.info(f"BVH dataloaders: train={len(self.train_data)}, test={len(self.test_data)}")
        else:
            logger.info(f"BVH dataloaders: train={len(self.train_data)}, test=skipped")

    def _init_smplx_dataloaders(self, cfg):
        """Initialize dataloaders for SMPL-X format (original)."""
        self.train_data = init_class(cfg.data.name_pyfile, cfg.data.class_name, cfg.data, loader_type='train')
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_data)
        self.train_loader = DataLoader(self.train_data, batch_size=cfg.data.train_bs, sampler=self.train_sampler, drop_last=True, num_workers=4)

        if cfg.data.test_clip:
            # test data for test_clip, only used for test_clip_fgd
            self.test_clip_data = init_class(cfg.data.name_pyfile, cfg.data.class_name, cfg.data, loader_type='test')
            self.test_clip_loader = DataLoader(self.test_clip_data, batch_size=64, drop_last=False)

        # test data for fgd, l1div and bc
        test_data_cfg = cfg.data.copy()
        test_data_cfg.test_clip = False
        self.test_data = init_class(cfg.data.name_pyfile, cfg.data.class_name, test_data_cfg, loader_type='test')
        self.test_loader = DataLoader(self.test_data, batch_size=1, drop_last=False)

    def _init_smplx_evaluation(self, cfg):
        """Initialize SMPL-X specific evaluation components."""
        eval_model_module = __import__(f"models.motion_representation", fromlist=["something"])
        eval_args = type('Args', (), {})()
        eval_args.vae_layer = 4
        eval_args.vae_length = 240
        eval_args.vae_test_dim = 330
        eval_args.variational = False
        eval_args.data_path_1 = "./datasets/hub/"
        eval_args.vae_grow = [1,1,2,1]

        eval_copy = getattr(eval_model_module, 'VAESKConv')(eval_args).to(self.rank)
        other_tools.load_checkpoints(
            eval_copy,
            './datasets/BEAT_SMPL/beat_v2.0.0/beat_english_v2.0.0/weights/AESKConv_240_100.bin',
            'VAESKConv'
        )
        self.eval_copy = eval_copy


        self.smplx = smplx.create(
            self.cfg.data.data_path_1+"smplx_models/",
            model_type='smplx',
            gender='NEUTRAL_2020',
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100,
            ext='npz',
            use_pca=False,
        ).to(self.rank).eval()

        self.alignmenter = metric.alignment(0.3, 7, self.train_data.avg_vel, upper_body=[3,6,9,12,13,14,15,16,17,18,19,20,21]) if self.rank == 0 else None
        self.align_mask = 60
        self.l1_calculator = metric.L1div() if self.rank == 0 else None

    def train_recording(self, epoch, its, t_data, t_train, mem_cost, lr_g, lr_d=None):
        """Enhanced training metrics logging"""
        metrics = {}
        
        # Collect all metrics
        for name, states in self.tracker.loss_meters.items():
            metric = states['train']
            if metric.count > 0:
                value = metric.avg
                metrics[name] = value
                
                metrics[f"train/{name}"] = value

        # Add learning rates and memory usage
        metrics.update({
            "train/learning_rate": lr_g,
            "train/data_time_ms": t_data*1000,
            "train/train_time_ms": t_train*1000,
        })
        

        # Log all metrics at once if using wandb
        wandb.log(metrics, step=epoch*self.train_length+its)

        # Print progress
        pstr = f"[{epoch:03d}][{its:03d}/{self.train_length:03d}]  "
        pstr += " ".join([f"{k}: {v:.3f}" for k, v in metrics.items() if "train/" not in k])
        logger.info(pstr)


    def val_recording(self, epoch):
        """Enhanced validation metrics logging"""
        metrics = {}
        
        # Process all validation metrics
        for name, states in self.tracker.loss_meters.items():
            metric = states['val']
            if metric.count > 0:
                value = float(metric.avg) if metric.count > 0 else float(metric.sum)
                metrics[f"val/{name}"] = value
                
                # Compare with best values to track best performance
                if name in self.val_best:
                    current_best = self.val_best[name]["value"]
                    # Custom comparison logic
                    if name in ["fgd", "test_clip_fgd"]:
                        is_better = value < current_best
                    elif name in ["l1div", "bc"]:
                        is_better = value > current_best
                    else:
                        is_better = value < current_best  # Default: lower is better

                    if is_better:
                        self.val_best[name] = {
                            "value": float(value),
                            "epoch": int(epoch)
                        }
                        
                        # Save best checkpoint separately
                        self.save_checkpoint(
                            epoch=epoch,
                            iteration=epoch * len(self.train_loader),
                            is_best=True,
                            best_metric_name=name
                        )
                    
                    # Add best value to metrics
                    metrics[f"best_{name}"] = float(self.val_best[name]["value"])
                    metrics[f"best_{name}_epoch"] = int(self.val_best[name]["epoch"])

        # Always save regular checkpoint for every validation
        self.save_checkpoint(
            epoch=epoch,
            iteration=epoch * len(self.train_loader),
            is_best=False,
            best_metric_name=None
        )

        # Log metrics
        if self.rank == 0:
            try:
                wandb.log(metrics, step=epoch*len(self.train_loader))
            except:
                logger.info("WANDB not initialized ! Probably doing the testing now")
        
        # Print validation results
        pstr = "Validation Results >>>> "
        pstr += " ".join([
            f"{k.split('/')[-1]}: {v:.3f}" 
            for k, v in metrics.items() 
            if k.startswith("val/")
        ])
        logger.info(pstr)

        # Print best results
        pstr = "Best Results >>>> "
        pstr += " ".join([
            f"{k}: {v['value']:.3f} (epoch {v['epoch']})" 
            for k, v in self.val_best.items()
        ])
        logger.info(pstr)

    def test_recording(self, dict_name, value, epoch):
        self.tracker.update_meter(dict_name, "test", value)
        _ = self.tracker.update_values(dict_name, 'test', epoch)

    def save_checkpoint(self, epoch, iteration, is_best=False, best_metric_name=None):
        """Save training checkpoint
        Args:
            epoch (int): Current epoch number
            iteration (int): Current iteration number
            is_best (bool): Whether this is the best model so far
            best_metric_name (str, optional): Name of the metric if this is a best checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'scheduler_state_dict': self.opt_s.state_dict() if hasattr(self, 'opt_s') and self.opt_s else None,
            'val_best': self.val_best,
        }
        
        # Save regular checkpoint every 20 epochs
        if epoch % 20 == 0:
            checkpoint_path = os.path.join(self.checkpoint_path, f"checkpoint_{epoch}")
            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(checkpoint, os.path.join(checkpoint_path, "ckpt.pth"))
        
        # Save best checkpoint if specified
        if is_best and best_metric_name:
            best_path = os.path.join(self.checkpoint_path, f"best_{best_metric_name}")
            os.makedirs(best_path, exist_ok=True)
            torch.save(checkpoint, os.path.join(best_path, "ckpt.pth"))

def prepare_all():
    """
    Parse command line arguments and prepare configuration
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/intention_w_distill.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true", help="Enable debugging mode")
    parser.add_argument("--mode", type=str, choices=['train', 'test', 'render'], default='train',
                       help="Choose between 'train' or 'test' or 'render' mode")
    parser.add_argument("--checkpoint", type=str, default=None, 
                       help="Checkpoint path for testing or resuming training")
    parser.add_argument('overrides', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Load config
    if args.config.endswith(".yaml"):
        cfg = OmegaConf.load(args.config)
        cfg.exp_name = args.config.split("/")[-1][:-5]
    else:
        raise ValueError("Unsupported config file format. Only .yaml files are allowed.")
    
    # Handle resume from checkpoint
    if args.resume:
        cfg.resume_from_checkpoint = args.resume
        
    # Debug mode settings
    if args.debug:
        cfg.wandb_project = "debug"
        cfg.exp_name = "debug"
        cfg.solver.max_train_steps = 4

    # Process override arguments
    if args.overrides:
        for arg in args.overrides:
            if '=' in arg:
                key, value = arg.split('=')
                try:
                    value = eval(value)
                except:
                    pass
                if key in cfg:
                    cfg[key] = value
                else:
                    try:
                        # Handle nested config with dot notation
                        keys = key.split('.')
                        cfg_node = cfg
                        for k in keys[:-1]:
                            cfg_node = cfg_node[k]
                        cfg_node[keys[-1]] = value
                    except:
                        raise ValueError(f"Key {key} not found in config.")
    
    # Set up wandb
    if hasattr(cfg, 'wandb_key'):
        os.environ["WANDB_API_KEY"] = cfg.wandb_key

    # Create output directories
    save_dir = os.path.join(cfg.output_dir, cfg.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'sanity_check'), exist_ok=True)

    # Save config
    config_path = os.path.join(save_dir, 'sanity_check', f'{cfg.exp_name}.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(cfg, f)

    # Copy source files for reproducibility
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sanity_check_dir = os.path.join(save_dir, 'sanity_check')
    output_dir = os.path.abspath(cfg.output_dir)
    
    def is_in_output_dir(path):
        return os.path.abspath(path).startswith(output_dir)
    
    def should_copy_file(file_path):
        if is_in_output_dir(file_path):
            return False
        if '__pycache__' in file_path:
            return False
        if file_path.endswith('.pyc'):
            return False
        return True

    # Copy Python files
    for root, dirs, files in os.walk(current_dir):
        if is_in_output_dir(root):
            continue
            
        for file in files:
            if file.endswith(".py"):
                full_file_path = os.path.join(root, file)
                if should_copy_file(full_file_path):
                    relative_path = os.path.relpath(full_file_path, current_dir)
                    dest_path = os.path.join(sanity_check_dir, relative_path)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    try:
                        shutil.copy(full_file_path, dest_path)
                    except Exception as e:
                        print(f"Warning: Could not copy {full_file_path}: {str(e)}")
    
    return cfg, args


def init_class(module_name, class_name, config, **kwargs):
    """
    Dynamically import and initialize a class
    """
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    instance = model_class(config, **kwargs)
    return instance

def seed_everything(seed):
    """
    Set random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@logger.catch
def main_worker(rank, world_size, cfg, args):
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
    logger_tools.set_args_and_logger(cfg, rank)
    seed_everything(cfg.seed)
    other_tools.print_exp_info(cfg)
      
    # Initialize trainer
    trainer = __import__(f"shortcut_rvqvae_trainer", fromlist=["something"]).CustomTrainer(cfg, args)
    
    # Resume logic
    resume_epoch = 0
    if args.resume:
        # Find the checkpoint path
        if os.path.isdir(args.resume):
            ckpt_path = os.path.join(args.resume, "ckpt.pth")
        else:
            ckpt_path = args.resume
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        trainer.load_checkpoint(checkpoint)
        resume_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
        logger.info(f"Resumed from checkpoint {ckpt_path}, starting at epoch {resume_epoch}")
    
    if args.mode == "train" and not args.resume:
        logger.info("Training from scratch ...")
    elif args.mode == "train" and args.resume:
        logger.info(f"Resuming training from checkpoint {args.resume} ...")
    elif args.mode == "test":
        logger.info("Testing ...")
    elif args.mode == "render":
        logger.info("Rendering ...")
    
    if args.mode == "train":
        start_time = time.time()
        for epoch in range(resume_epoch, cfg.solver.epochs+1):
            if cfg.ddp: 
                trainer.val_loader.sampler.set_epoch(epoch)
            
            
            if (epoch) % cfg.val_period == 0 and epoch > 0:
                if rank == 0:
                    if cfg.data.test_clip:
                        trainer.test_clip(epoch)
                    else:
                        trainer.val(epoch)
            
            epoch_time = time.time()-start_time
            if trainer.rank == 0: 
                logger.info(f"Time info >>>> elapsed: {epoch_time/60:.2f} mins\t" + 
                        f"remain: {(cfg.solver.epochs/(epoch+1e-7)-1)*epoch_time/60:.2f} mins")
            
            if epoch != cfg.solver.epochs:
                if cfg.ddp: 
                    trainer.train_loader.sampler.set_epoch(epoch)
                trainer.tracker.reset()
                trainer.train(epoch)
                
            if cfg.debug:
                trainer.test(epoch)
                
            
        
        # Final cleanup and logging
        if rank == 0:
            for k, v in trainer.val_best.items():
                logger.info(f"Best {k}: {v['value']:.6f} at epoch {v['epoch']}")
            
            wandb.finish()
    elif args.mode == "test":
        trainer.test_clip(999)
        trainer.test(999)
    elif args.mode == "render":
        trainer.test_render(999)

if __name__ == "__main__":
    # Set up distributed training environment
    master_addr = '127.0.0.1'
    master_port = 29500
    
    import socket
    # Function to check if a port is in use
    def is_port_in_use(port, host='127.0.0.1'):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return False  # Port is available
            except socket.error:
                return True   # Port is in use
    
    # Find available port
    while is_port_in_use(master_port):
        print(f"Port {master_port} is in use, trying next port...")
        master_port += 1
    
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    
    cfg, args = prepare_all()
    
    if cfg.ddp:
        mp.set_start_method("spawn", force=True)
        mp.spawn(
            main_worker,
            args=(len(cfg.gpus), cfg, args),
            nprocs=len(cfg.gpus),
        )
    else:
        main_worker(0, 1, cfg, args)