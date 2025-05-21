"""Adapted from TODO"""

import argparse
import json
import os
import re
from datetime import datetime

import torch
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from diffusion import IDDPM, DPMS, SASolverSampler
from diffusion.utils.misc import set_random_seed, read_config
from diffusion.model.utils import prepare_prompt_ar
from diffusion.model.builder import build_model
from diffusion.model.t5 import T5Embedder
from diffusion.data.datasets import get_chunks, ASPECT_RATIO_512_TEST, ASPECT_RATIO_1024_TEST, ASPECT_RATIO_256_TEST
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from tools.download import find_model

from mup import set_base_shapes

torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default='./', help="root")
    parser.add_argument("--metadata_file", type=str, default="tools/geneval-main/prompts/evaluation_metadata.jsonl", help="JSONL file containing lines of metadata for each prompt")
    parser.add_argument("--config", type=str, default='configs/pixart_config/PixArt_mup_xl2_img256_SAM_proxy.py', help="config")
    parser.add_argument('--load_base_shapes', type=str, default='', help='file location to load base shapes from')
    parser.add_argument('--pretrained_models_dir', default='output/pretrained_models', type=str)
    parser.add_argument('--model_path', default='output/search_SAM_256/loglr-10/checkpoints/epoch_5_step_39185.pth', type=str)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--sampling_algo', default='dpm-solver', type=str, choices=['iddpm', 'dpm-solver', 'sa-solver'])
    parser.add_argument('--step', default=-1, type=int)
    parser.add_argument('--cfg_scale', default=4.5, type=float)
    parser.add_argument("--n_samples", type=int, default=4, help="number of samples")
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible sampling)")
    parser.add_argument("--batch_size", type=int, default=4, help="how many samples can be produced simultaneously")
    parser.add_argument("--skip_grid", action="store_true", help="skip saving grid")
    opt = parser.parse_args()
    return opt


def main(opt):
    opt.config = os.path.join(opt.root, opt.config)
    opt.model_path = os.path.join(opt.root, opt.model_path)
    opt.pretrained_models_dir = os.path.join(opt.root, opt.pretrained_models_dir)
    opt.metadata_file = os.path.join(opt.root, opt.metadata_file)

    # Load prompts
    with open(opt.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    # Load model
    config = read_config(opt.config)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    assert opt.sampling_algo in ['iddpm', 'dpm-solver', 'sa-solver']

    # only support fixed latent size currently
    latent_size = opt.image_size // 8
    sample_steps_dict = {'iddpm': 100, 'dpm-solver': 20, 'sa-solver': 25}
    sample_steps = opt.step if opt.step != -1 else sample_steps_dict[opt.sampling_algo]
    weight_dtype = torch.float
    print(f"Inference with {weight_dtype}")

    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    model_kwargs={"window_block_indexes": config.window_block_indexes, "window_size": config.window_size,
                  "use_rel_pos": config.use_rel_pos, "lewei_scale": config.lewei_scale,
                  'model_max_length': config.model_max_length}

    # model setting
    model = build_model(config.model,
            config.grad_checkpointing,
            config.get('fp32_attention', False),
            num_heads=config.head,
            hidden_size=config.head*72,
            mup=config.mup,
            input_size=latent_size,
            learn_sigma=learn_sigma,
            pred_sigma=pred_sigma,
            **model_kwargs)
    if config.mup:
        opt.load_base_shapes = os.path.join(opt.root, opt.load_base_shapes)
        set_base_shapes(model, opt.load_base_shapes)
    model = model.to(device)

    print(f"Generating sample from ckpt: {opt.model_path}")
    state_dict = find_model(opt.model_path)
    del state_dict['state_dict']['pos_embed']
    missing, unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)
    print('Missing keys: ', missing)
    print('Unexpected keys', unexpected)
    model.eval()
    model.to(weight_dtype)
    base_ratios = eval(f'ASPECT_RATIO_{opt.image_size}_TEST')

    vae = AutoencoderKL.from_pretrained(f'{opt.pretrained_models_dir}/sd-vae-ft-ema').to(device)
    t5 = T5Embedder(device=device, local_cache=True, cache_dir=f'{opt.pretrained_models_dir}', dir_or_name='t5-v1_1-xxl', model_max_length=120)
    work_dir = os.path.join(*opt.model_path.split('/')[:-2])
    work_dir = f'/{work_dir}' if opt.model_path[0] == '/' else work_dir

    # img save setting
    try:
        epoch_name = re.search(r'.*epoch_(\d+).*.pth', opt.model_path).group(1)
        step_name = re.search(r'.*step_(\d+).*.pth', opt.model_path).group(1)
    except Exception:
        epoch_name = 'unknown'
        step_name = 'unknown'
    img_save_dir = os.path.join(work_dir, 'vis')
    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(img_save_dir, exist_ok=True)

    save_root = os.path.join(img_save_dir, f"{datetime.now().date()}_geneval_epoch{epoch_name}_step{step_name}_scale{opt.cfg_scale}_step{sample_steps}_size{opt.image_size}_bs{opt.batch_size}_samp{opt.sampling_algo}_seed{opt.seed}")
    os.makedirs(save_root, exist_ok=True)

    for index, metadata in enumerate(metadatas):
        set_random_seed(opt.seed)

        outpath = os.path.join(save_root, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        prompt = metadata['prompt']
        n_rows = batch_size = opt.batch_size
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        sample_count = 0

        with torch.no_grad():
            all_samples = list()
            for n in trange((opt.n_samples + batch_size - 1) // batch_size, desc="Sampling"):
                # Generate images
                prompts = []
                if batch_size == 1:
                    prompt_clean, _, hw, ar, custom_hw = prepare_prompt_ar(prompt, base_ratios, device=device, show=False)  # ar for aspect ratio
                    hw = torch.tensor([[opt.image_size, opt.image_size]], dtype=torch.float, device=device).repeat(batch_size, 1)
                    ar = torch.tensor([[1.]], device=device).repeat(batch_size, 1)
                    latent_size_h, latent_size_w = latent_size, latent_size
                    prompts.append(prompt_clean.strip())
                else:
                    hw = torch.tensor([[opt.image_size, opt.image_size]], dtype=torch.float, device=device).repeat(batch_size, 1)
                    ar = torch.tensor([[1.]], device=device).repeat(batch_size, 1)
                    latent_size_h, latent_size_w = latent_size, latent_size
                    for _ in range(batch_size):
                        prompts.append(prepare_prompt_ar(prompt, base_ratios, device=device, show=False)[0].strip())

                null_y = model.y_embedder.y_embedding[None].repeat(len(prompts), 1, 1)[:, None]

                caption_embs, emb_masks = t5.get_text_embeddings(prompts)
                caption_embs = caption_embs.float()[:, None]
                print('finish embedding')

                if opt.sampling_algo == 'iddpm':
                    # Create sampling noise:
                    n = len(prompts)
                    z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device).repeat(2, 1, 1, 1)
                    model_kwargs = dict(y=torch.cat([caption_embs, null_y]),
                                        cfg_scale=opt.cfg_scale, data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                    diffusion = IDDPM(str(sample_steps))
                    # Sample images:
                    samples = diffusion.p_sample_loop(
                        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                        device=device
                    )
                    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                elif opt.sampling_algo == 'dpm-solver':
                    # Create sampling noise:
                    n = len(prompts)
                    z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device)
                    model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                    dpm_solver = DPMS(model.forward_with_dpmsolver,
                                    condition=caption_embs,
                                    uncondition=null_y,
                                    cfg_scale=opt.cfg_scale,
                                    model_kwargs=model_kwargs)
                    samples = dpm_solver.sample(
                        z,
                        steps=sample_steps,
                        order=2,
                        skip_type="time_uniform",
                        method="multistep",
                    )
                elif opt.sampling_algo == 'sa-solver':
                    # Create sampling noise:
                    n = len(prompts)
                    model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                    sa_solver = SASolverSampler(model.forward_with_dpmsolver, device=device)
                    samples = sa_solver.sample(
                        S=25,
                        batch_size=n,
                        shape=(4, latent_size_h, latent_size_w),
                        eta=1,
                        conditioning=caption_embs,
                        unconditional_conditioning=null_y,
                        unconditional_guidance_scale=opt.cfg_scale,
                        model_kwargs=model_kwargs,
                    )[0]

                samples = vae.decode(samples / 0.18215).sample
                torch.cuda.empty_cache()

                # Save images
                os.umask(0o000)  # file permission: 666; dir permission: 777
                for sample in samples:
                    save_path = os.path.join(sample_path, f"{sample_count:05}.png")
                    print("Saving path: ", save_path)
                    save_image(sample, save_path, nrow=1, normalize=True, value_range=(-1, 1))
                    sample_count += 1
                if not opt.skip_grid:
                    all_samples.append(torch.stack([sample for sample in samples], 0))

            if not opt.skip_grid:
                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                grid = Image.fromarray(grid.astype(np.uint8))
                grid.save(os.path.join(outpath, f'grid.png'))
                del grid
        del all_samples

    print("Done.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
