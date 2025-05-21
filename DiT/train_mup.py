# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True # TensorFloat32 (TF32)
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch.optim import AdamW
from mup.optim import MuAdamW
from mup import get_shapes, set_base_shapes, make_base_shapes

from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image

import numpy as np
from collections import OrderedDict
from copy import deepcopy
from time import time
import argparse
import logging
import os
import json
from utils import seed_torch

from models_mup import DiT_mup
from models import DiT_sp
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

import sys # for debugging


# Training Helper Functions

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        # model_weights = decay * model_weights + (1 - decay) * new_model_weights
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


# Training Loop

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Variables for monitoring/logging purposes:
    df_loss = []
    start_epoch = 0
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    seed_torch(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        args.results_dir = os.path.join(args.root, args.results_dir)
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        # experiment_index = len(glob(f"{args.results_dir}/*"))
        if args.mup:
            experiment_name = f'muP_DiT_L{args.depth}_p{args.patch_size}_d{args.dim_heads}_h{args.num_heads}_loglr{args.loglr}_logstd{args.logstd}_B{args.global_batch_size}'
        else:
            experiment_name = f'SP_DiT_L{args.depth}_p{args.patch_size}_d{args.dim_heads}_h{args.num_heads}_loglr{args.loglr}_logstd{args.logstd}_B{args.global_batch_size}'
        experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        with open(os.path.join(experiment_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)
    else:
        logger = create_logger(None)

    # Create model:
    hidden_size = args.dim_heads * args.num_heads
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    
    if args.mup:
        output_mult = 2 ** args.logstd
        model = DiT_mup(
            depth=args.depth, hidden_size=hidden_size, patch_size=args.patch_size, num_heads=args.num_heads,
            input_size=latent_size, num_classes=args.num_classes, output_mult=output_mult
        )
        assert args.load_base_shapes, 'load_base_shapes needs to be nonempty'
        args.load_base_shapes = os.path.join(args.root, args.load_base_shapes)
        set_base_shapes(model, args.load_base_shapes)
    else:
        model = DiT_sp(
            depth=args.depth, hidden_size=hidden_size, patch_size=args.patch_size, num_heads=args.num_heads,
            input_size=latent_size, num_classes=args.num_classes
        )
        # set_base_shapes(model, None)
    model.to(device)
    
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    lr = 2 ** args.loglr
    if args.mup:
        # mup: use muadamw to adjust the layer-wise learning rate.
        opt = MuAdamW(model.parameters(), lr=lr, weight_decay=0)
    else:
        opt = AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    
    # Load checkpoint
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'], strict=True)
        ema.load_state_dict(checkpoint['ema'], strict=True)
        opt.load_state_dict(checkpoint['opt'])
        del checkpoint
        logger.info(f"Using checkpoint: {args.ckpt}")
    
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae_path = os.path.join(args.root, f"vaes/sd-vae-ft-{args.vae}")
    vae = AutoencoderKL.from_pretrained(vae_path, local_files_only=True).to(device)

    model = DDP(model, device_ids=[device])
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    args.data_path = os.path.join(args.root, args.data_path)
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Initial state
    if args.ckpt:
        train_steps = int(args.ckpt.split('/')[-1].split('.')[0])
        start_epoch = int(train_steps / (len(dataset) / args.global_batch_size))
        logger.info(f"Initial state: step={train_steps}, epoch={start_epoch}")
    else:
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                df_loss.append(avg_loss)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

                    save_path_loss = os.path.join(experiment_dir, 'log_loss.npy')
                    if os.path.exists(save_path_loss):
                        os.remove(save_path_loss)
                    np.save(save_path_loss, np.array(df_loss))
                
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./")
    parser.add_argument("--data-path", type=str, default="path/imagenet/train")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument('--save_base_shapes', type=str, default='', help='file location to save base shapes at')
    parser.add_argument('--load_base_shapes', type=str, default='', help='file location to load base shapes from')
    parser.add_argument("--global-seed", type=int, default=0)

    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument('--mup', action='store_true', help='if use mup.')
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--dim_heads", type=int, default=72)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument('--logstd', type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training

    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument('--loglr', type=float, default=-10)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)

    parser.add_argument("--ckpt", type=str, default=None)
    
    args = parser.parse_args()

    if args.save_base_shapes != '':
        print(f'saving base shapes at {args.save_base_shapes}')
        base_model = DiT_mup(
            depth=28, hidden_size=288, patch_size=2, num_heads=4,
            input_size=32, num_classes=1000
        )
        base_shapes = get_shapes(base_model)

        delta_model = DiT_mup(
            depth=28, hidden_size=360, patch_size=2, num_heads=5,
            input_size=32, num_classes=1000
        )
        delta_shapes = get_shapes(delta_model)
        
        make_base_shapes(base_shapes, delta_shapes, savefile=args.save_base_shapes)
        print('done and exit')
        sys.exit()

    main(args)
