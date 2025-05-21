"""
python clip_score.py --image_dir /path/to/images --batch_size 512 --model_name ViT-B/32
"""

import os
import torch
import torch.nn.functional as F
import clip
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir, preprocess):
        self.image_paths = [
            os.path.join(image_dir, f) 
            for f in os.listdir(image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.preprocess = preprocess
        self.caption_dict = {
            os.path.basename(p): os.path.splitext(os.path.basename(p))[0]
            for p in self.image_paths
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = self.preprocess(Image.open(path))
        except:
            image = Image.new("RGB", (224, 224))
        caption = self.caption_dict[os.path.basename(path)]
        return image, caption, os.path.basename(path)

def calculate_clip_score(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.model_name, device=device)
    model = model.to(device).eval()
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    dataset = ImageCaptionDataset(args.image_dir, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    results = []
    with torch.no_grad():
        for images, texts, filenames in tqdm(dataloader, desc="Processing batches"):

            images = images.to(device)
            text_inputs = clip.tokenize(texts, truncate=True).to(device)

            with torch.no_grad():
                image_features = model.encode_image(images)
                text_features = model.encode_text(text_inputs)
                # print('model.logit_scale.exp(): ', model.logit_scale.exp())
                scores = F.cosine_similarity(image_features, text_features, dim=1) * model.logit_scale.exp()

            batch_results = [
                (filename, score.item()) for filename, score in zip(filenames, scores)
            ]
            results.extend(batch_results)

    df = pd.DataFrame(results, columns=["filename", "clip_score"])
    df.to_csv(args.save_path, index=False)
    
    print(f"\nCLIP Score Statistics (n={len(df)}):")
    print(f"Mean: {df['clip_score'].mean():.4f}")
    print(f"Std : {df['clip_score'].std():.4f}")
    print(f"Min : {df['clip_score'].min():.4f}")
    print(f"Max : {df['clip_score'].max():.4f}")
    print(f"Results saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate CLIP Scores for images")
    parser.add_argument("--image_dir", type=str, required=True,
                       help="Directory containing image files")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size for processing (default: 256)")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of data loading workers (default: 8)")
    parser.add_argument("--model_name", type=str, default="ViT-B/32",
                       choices=["RN50", "RN101", "RN50x4", "ViT-B/32"],
                       help="CLIP model variant (default: ViT-B/32)")
    parser.add_argument("--save_path", type=str, default="clip_scores.csv",
                       help="Output CSV file path (default: clip_scores.csv)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_dir):
        raise FileNotFoundError(f"Image directory {args.image_dir} does not exist!")
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    calculate_clip_score(args)