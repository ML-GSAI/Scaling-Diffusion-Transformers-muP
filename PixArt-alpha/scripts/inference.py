import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import warnings
warnings.filterwarnings("ignore")  # ignore warning
import re
import argparse
from datetime import datetime
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL

from diffusion.model.utils import prepare_prompt_ar
from diffusion.model.builder import build_model
from diffusion import IDDPM, DPMS, SASolverSampler
from tools.download import find_model
from diffusion.model.t5 import T5Embedder
from diffusion.data.datasets import get_chunks, ASPECT_RATIO_512_TEST, ASPECT_RATIO_1024_TEST, ASPECT_RATIO_256_TEST
from diffusion.utils.misc import read_config

from mup import set_base_shapes

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default='./', help="root")
    parser.add_argument("--config", type=str, default='configs/pixart_config/PixArt_mup_xl2_img256_SAM_proxy.py', help="config")
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--pretrained_models_dir', default='output/pretrained_models', type=str)
    parser.add_argument('--model_path', default='output/search_SAM_256/loglr-10/checkpoints/epoch_5_step_39185.pth', type=str)
    parser.add_argument('--bs', default=50, type=int)
    parser.add_argument('--cfg_scale', default=4.5, type=float)
    parser.add_argument('--sampling_algo', default='dpm-solver', type=str, choices=['iddpm', 'dpm-solver', 'sa-solver'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset', type=str, default='mscoco', help='dataset')
    parser.add_argument('--step', default=-1, type=int)
    parser.add_argument('--save_name', default='test_sample', type=str)
    parser.add_argument('--load_base_shapes', type=str, default='', help='file location to load base shapes from')

    return parser.parse_args()


def set_env(seed=0):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, args.image_size, args.image_size)


@torch.inference_mode()
def visualize(items, bs, sample_steps, cfg_scale):

    for chunk in tqdm(list(get_chunks(items, bs)), unit='batch'):

        prompts = []
        if bs == 1:
            prompt_clean, _, hw, ar, custom_hw = prepare_prompt_ar(chunk[0], base_ratios, device=device, show=False)  # ar for aspect ratio
            if args.image_size == 1024:
                latent_size_h, latent_size_w = int(hw[0, 0] // 8), int(hw[0, 1] // 8)
            else:
                hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
                ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
                latent_size_h, latent_size_w = latent_size, latent_size
            prompts.append(prompt_clean.strip())
        else:
            hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
            ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
            for prompt in chunk:
                prompts.append(prepare_prompt_ar(prompt, base_ratios, device=device, show=False)[0].strip())
            latent_size_h, latent_size_w = latent_size, latent_size

        null_y = model.y_embedder.y_embedding[None].repeat(len(prompts), 1, 1)[:, None]

        with torch.no_grad():
            caption_embs, emb_masks = t5.get_text_embeddings(prompts)
            caption_embs = caption_embs.float()[:, None]
            print('finish embedding')

            if args.sampling_algo == 'iddpm':
                # Create sampling noise:
                n = len(prompts)
                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device).repeat(2, 1, 1, 1)
                model_kwargs = dict(y=torch.cat([caption_embs, null_y]),
                                    cfg_scale=cfg_scale, data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                diffusion = IDDPM(str(sample_steps))
                # Sample images:
                samples = diffusion.p_sample_loop(
                    model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                    device=device
                )
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            elif args.sampling_algo == 'dpm-solver':
                # Create sampling noise:
                n = len(prompts)
                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device)
                model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                dpm_solver = DPMS(model.forward_with_dpmsolver,
                                  condition=caption_embs,
                                  uncondition=null_y,
                                  cfg_scale=cfg_scale,
                                  model_kwargs=model_kwargs)
                samples = dpm_solver.sample(
                    z,
                    steps=sample_steps,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                )
            elif args.sampling_algo == 'sa-solver':
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
                    unconditional_guidance_scale=cfg_scale,
                    model_kwargs=model_kwargs,
                )[0]
        samples = vae.decode(samples / 0.18215).sample
        torch.cuda.empty_cache()
        # Save images:
        os.umask(0o000)  # file permission: 666; dir permission: 777
        for i, sample in enumerate(samples):
            if args.dataset == 'mscoco':
                img_name = re.sub(r'[\\/*?:"<>|.]', '_', prompts[i][:100])
            elif args.dataset =='mjhq':
                img_name = re.sub(r'[\\/*?:"<>|.]', '_', prompts[i][:200])
            save_path = os.path.join(save_root, f"{img_name}.jpg")
            print("Saving path: ", save_path)
            save_image(sample, save_path, nrow=1, normalize=True, value_range=(-1, 1))


if __name__ == '__main__':
    args = get_args()

    args.config = os.path.join(args.root, args.config)
    config = read_config(args.config)

    # Setup PyTorch:
    seed = args.seed
    set_env(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert args.sampling_algo in ['iddpm', 'dpm-solver', 'sa-solver']

    # only support fixed latent size currently
    latent_size = args.image_size // 8
    sample_steps_dict = {'iddpm': 100, 'dpm-solver': 20, 'sa-solver': 25}
    sample_steps = args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]
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
        args.load_base_shapes = os.path.join(args.root, args.load_base_shapes)
        set_base_shapes(model, args.load_base_shapes)
    model = model.to(device)

    print(f"Generating sample from ckpt: {args.model_path}")
    state_dict = find_model(args.model_path)
    del state_dict['state_dict']['pos_embed']
    missing, unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)
    print('Missing keys: ', missing)
    print('Unexpected keys', unexpected)
    model.eval()
    model.to(weight_dtype)
    base_ratios = eval(f'ASPECT_RATIO_{args.image_size}_TEST')

    vae = AutoencoderKL.from_pretrained(f'{args.pretrained_models_dir}/sd-vae-ft-ema').to(device)
    t5 = T5Embedder(device=device, local_cache=True, cache_dir=f'{args.pretrained_models_dir}', dir_or_name='t5-v1_1-xxl', model_max_length=120)
    work_dir = os.path.join(*args.model_path.split('/')[:-2])
    work_dir = f'/{work_dir}' if args.model_path[0] == '/' else work_dir

    # data setting
    if args.dataset == 'mscoco':
        args.txt_file = 'data/mscoco/captions.txt'
    elif args.dataset =='mjhq':
        args.txt_file = 'data/mjhq/captions.txt'
    with open(args.txt_file, 'r') as f:
        items = [item.strip() for item in f.readlines()]

    # img save setting
    try:
        epoch_name = re.search(r'.*epoch_(\d+).*.pth', args.model_path).group(1)
        step_name = re.search(r'.*step_(\d+).*.pth', args.model_path).group(1)
    except Exception:
        epoch_name = 'unknown'
        step_name = 'unknown'
    img_save_dir = os.path.join(work_dir, 'vis')
    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(img_save_dir, exist_ok=True)

    save_root = os.path.join(img_save_dir, f"{datetime.now().date()}_{args.dataset}_epoch{epoch_name}_step{step_name}_scale{args.cfg_scale}_step{sample_steps}_size{args.image_size}_bs{args.bs}_samp{args.sampling_algo}_seed{seed}")
    os.makedirs(save_root, exist_ok=True)
    visualize(items, args.bs, sample_steps, args.cfg_scale)