import os
import numpy as np

import torch

import argparse

from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.model.builder import build_model
from diffusion.model.nets import PixArt_ANY
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow


from tools.coord_check import get_coord_data, plot_coord_data
from mup import get_shapes, set_base_shapes, make_base_shapes

import sys

def coord_check(mup, lr, train_loader, nsteps, nseeds, args, config, plotdir='', legend=False):

    def gen(head, config, standparam=False):
        def f():
            pred_sigma = getattr(config, 'pred_sigma', True)
            learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
            model_kwargs={"window_block_indexes": config.window_block_indexes, "window_size": config.window_size,
                  "use_rel_pos": config.use_rel_pos, "lewei_scale": config.lewei_scale, 'config':config,
                  'model_max_length': config.model_max_length}
            
            if standparam:
                model = build_model(config.model,
                        config.grad_checkpointing,
                        config.get('fp32_attention', False),
                        num_heads=head,
                        hidden_size=head*72,
                        mup=False,
                        input_size=32,
                        learn_sigma=learn_sigma,
                        pred_sigma=pred_sigma,
                        **model_kwargs)
                set_base_shapes(model, None)
            else:
                assert args.load_base_shapes, 'load_base_shapes needs to be nonempty'
                model = build_model(config.model,
                        config.grad_checkpointing,
                        config.get('fp32_attention', False),
                        num_heads=head,
                        hidden_size=head*72,
                        mup=True,
                        input_size=32,
                        learn_sigma=learn_sigma,
                        pred_sigma=pred_sigma,
                        **model_kwargs)
                set_base_shapes(model, args.load_base_shapes)
                model.initialize_weights(mup_init=True)
            return model.to(device)
        return f

    heads = np.arange(1, 20, 3)
    models = {w*72: gen(w, config, standparam=not mup) for w in heads}
    opt = 'adamw'

    df = get_coord_data(models, train_loader, config=config, mup=mup, lr=lr, optimizer=opt, nseeds=nseeds, nsteps=nsteps)

    prm = 'muP' if mup else 'SP'
    return plot_coord_data(df, legend=legend,save_to=os.path.join(plotdir, f'{prm}_PixArt_{opt}_coord.png'),suptitle=f'{prm} Pixelart {opt} lr={lr} nseeds={nseeds}',face_color='xkcd:light grey' if not mup else None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pixelart coordcheck', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--root", type=str, default='./', help="config")
    parser.add_argument("--config", type=str, default='configs/pixart_config/PixArt_mup_img256_SAM_coord.py', help="config")
    parser.add_argument('--work-dir', default='output/pixelart_coordcheck', help='the dir to save logs and models')
    
    parser.add_argument('--save_base_shapes', type=str, default='',
                        help='file location to save base shapes at')
    parser.add_argument('--load_base_shapes', type=str, default='',
                        help='file location to load base shapes from')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--coord_check_nsteps', type=int, default=4,
                        help='Do coord check with this many steps.')
    parser.add_argument('--coord_check_nseeds', type=int, default=3,
                        help='number of seeds for testing correctness of μ parametrization')

    args = parser.parse_args()

    args.config = os.path.join(args.root, args.config)
    config = read_config(args.config)
    assert args.work_dir is not None
    # update configs according to CLI args if args.work_dir is not None
    args.work_dir = os.path.join(args.root, args.work_dir)
    config.work_dir = args.work_dir
    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    device = torch.device("cuda")

    config.seed = init_random_seed(config.get('seed', None))
    set_random_seed(config.seed)

    if args.save_base_shapes:
        print(f'saving base shapes at {args.save_base_shapes}')
        base_shapes = get_shapes(PixArt_ANY(depth=28, hidden_size=72*4, num_heads=4, patch_size=2))
        delta_shapes = get_shapes(PixArt_ANY(depth=28, hidden_size=72*5, num_heads=5, patch_size=2))
        make_base_shapes(base_shapes, delta_shapes, savefile=args.save_base_shapes)
        print('done and exit')
        sys.exit()

    # Setup data:
    set_data_root(config.data_root)
    dataset = build_dataset(config.data, resolution=config.image_size, aspect_ratio_type=config.aspect_ratio_type)
    loader = build_dataloader(dataset, num_workers=config.num_workers, batch_size=config.train_batch_size, shuffle=True)
    print(f"Dataset contains {len(dataset):,} images")
    
    print('testing parametrization')
    coord_check(mup=config.mup, lr=args.lr, train_loader=loader, nsteps=args.coord_check_nsteps, nseeds=args.coord_check_nseeds, args=args, config=config, plotdir=config.work_dir, legend=False)
    sys.exit()
