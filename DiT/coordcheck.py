import os
import numpy as np

import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import argparse

from models_mup import DiT_mup
from models import DiT_sp
from train_mup import center_crop_arr

from utils.coord_check import get_coord_data, plot_coord_data
from mup import get_shapes, set_base_shapes, make_base_shapes

import sys

def coord_check(mup, lr, train_loader, nsteps, nseeds, args, plotdir='', legend=False):

    def gen(head, standparam=False):
        def f():
            if standparam:
                model = DiT_sp(depth=28, hidden_size=head*72, num_heads=head, patch_size=2)
                # set_base_shapes(model, None)
            else:
                assert args.load_base_shapes, 'load_base_shapes needs to be nonempty'
                model = DiT_mup(depth=28, hidden_size=head*72, num_heads=head, patch_size=2)
                set_base_shapes(model, args.load_base_shapes)
            return model.to(device)
        return f

    heads = np.arange(1, 20, 3)
    models = {w*72: gen(w, standparam=not mup) for w in heads}
    opt = 'adamw'

    df = get_coord_data(models, train_loader, mup=mup, lr=lr, optimizer=opt, nseeds=nseeds, nsteps=nsteps)

    prm = 'muP' if mup else 'SP'
    return plot_coord_data(df, legend=legend,save_to=os.path.join(plotdir, f'{prm}_DiT_{opt}_coord.png'),suptitle=f'{prm} DiT {opt} lr={lr} nseeds={nseeds}',face_color='xkcd:light grey' if not mup else None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''DiT mup coordcheck''', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('./', help="root")
    parser.add_argument('--work_dir', default='results/dit_coordcheck', help='the dir to save logs and models')
    
    parser.add_argument('--save_base_shapes', type=str, default='',
                        help='file location to save base shapes at')
    parser.add_argument('--load_base_shapes', type=str, default='',
                        help='file location to load base shapes from')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data_path', type=str, default='./data/imagenet/train')
    parser.add_argument('--coord_check_nsteps', type=int, default=4,
                        help='Do coord check with this many steps.')
    parser.add_argument('--coord_check_nseeds', type=int, default=3,
                        help='number of seeds for testing correctness of μ parametrization')
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)

    args = parser.parse_args()

    assert args.work_dir is not None
    # update configs according to CLI args if args.work_dir is not None
    args.work_dir = os.path.join(args.root, args.work_dir)
    os.umask(0o000)
    os.makedirs(args.work_dir, exist_ok=True)

    torch.manual_seed(args.seed)

    device = torch.device("cuda")

    if args.save_base_shapes != '':
        print(f'saving base shapes at {args.save_base_shapes}')
        base_shapes = get_shapes(DiT_mup(depth=28, hidden_size=72*4, num_heads=4, patch_size=2))
        delta_shapes = get_shapes(DiT_mup(depth=28, hidden_size=72*5, num_heads=5, patch_size=2))
        make_base_shapes(base_shapes, delta_shapes, savefile=args.save_base_shapes)
        print('done and exit')
        sys.exit()

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    print(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    
    print('testing parametrization')
    coord_check(mup=True, lr=args.lr, train_loader=loader, nsteps=args.coord_check_nsteps, nseeds=args.coord_check_nseeds, args=args, plotdir=args.work_dir, legend=False)
    # coord_check(mup=False, lr=args.lr, train_loader=loader, nsteps=args.coord_check_nsteps, nseeds=args.coord_check_nseeds, args=args, plotdir=plotdir, legend=True)
    sys.exit()
