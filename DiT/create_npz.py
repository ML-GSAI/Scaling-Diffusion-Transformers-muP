from tqdm import tqdm
import numpy as np
from PIL import Image
import os

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def main():
    root = "./results"
    para = 'muP'
    h_list = [4]
    loglr_list = [-9.0, -10.0, -11.0, -12.0, -13.0]
    logstd_list = [-2, 2, -4, 0, 4, -6, 6]
    cfg_list = [1.0]
    vae = 'mse'
    step_list = [200_000]
    for h in h_list:
        for loglr in loglr_list:
            for logstd in logstd_list:
                for cfg in cfg_list:
                    for step in step_list:
                        sample_folder_dir = f"{root}/{para}_DiT_L28_p2_d72_h{h}_loglr{loglr}_logstd{logstd}_B128/samples/{step:07d}_vae{vae}_cfg{cfg}"
                        if os.path.exists(sample_folder_dir) and not os.path.exists(f"{sample_folder_dir}.npz"):
                            print('begin create npz:', sample_folder_dir)
                            create_npz_from_sample_folder(sample_folder_dir)
                        print("Done.")

if __name__ == "__main__":
    main()