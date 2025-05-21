# GigaGAN: https://github.com/mingukkang/GigaGAN
# The MIT License (MIT)
# See license file or visit https://github.com/mingukkang/GigaGAN for details

# evaluation.py
import torch
import numpy as np

from pathlib import Path
from PIL import Image

from data_util import CenterCropLongEdge
from cleanfid import fid



def tensor2pil(image: torch.Tensor):
    ''' output image : tensor to PIL
    '''
    if isinstance(image, list) or image.ndim == 4:
        return [tensor2pil(im) for im in image]

    assert image.ndim == 3
    output_image = Image.fromarray(((image + 1.0) * 127.5).clamp(
        0.0, 255.0).to(torch.uint8).permute(1, 2, 0).detach().cpu().numpy())
    return output_image


@torch.no_grad()
def compute_fid(fake_dir: Path, gt_dir: Path,
    resize_size=None, feature_extractor="clip"):
    center_crop_trsf = CenterCropLongEdge()
    def resize_and_center_crop(image_np):
        image_pil = Image.fromarray(image_np) 
        image_pil = center_crop_trsf(image_pil)

        if resize_size is not None:
            image_pil = image_pil.resize((resize_size, resize_size),
                                         Image.LANCZOS)
        return np.array(image_pil)

    if feature_extractor == "inception":
        model_name = "inception_v3"
    elif feature_extractor == "clip":
        model_name = "clip_vit_b_32"
    else:
        raise ValueError("Unrecognized feature extractor [%s]" % feature_extractor)
    fidx = fid.compute_fid(gt_dir,
                          fake_dir,
                          model_name=model_name,
                          custom_image_tranform=resize_and_center_crop)
    return fidx


def evaluate_model(opt):
    ### Generated images
    fid = compute_fid(
        opt.ref_dir,
        opt.fake_dir,
        resize_size=opt.eval_res,
        feature_extractor="inception")
    print(f"FID_{opt.eval_res}px: {fid}")
    return


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir",
                        default="data/mscoco/coco_30k_randomly_sampled_2014_val",
                        help="location of the reference images for evaluation")
    parser.add_argument("--fake_dir",
                        default="",
                        help="location of fake images for evaluation")
    parser.add_argument("--eval_res", default=256, type=int)
    
    opt, _ = parser.parse_known_args()
    evaluate_model(opt)