import argparse
import torch
from cp_dataset import CPDataset
from networks import GMM  # Your fixed GMM model
from torch.utils.data import DataLoader
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="data")
    parser.add_argument("--datamode", type=str, default="test")
    parser.add_argument("--test_pairs", type=str, default="test_pairs.txt")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--result_dir", type=str, default="result")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    opt = parser.parse_args()

    os.makedirs(opt.result_dir, exist_ok=True)

    # Dataset
    dataset = CPDataset(opt)
    dataset = [d for d in dataset if d is not None]  # remove missing files
    data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    # Model
    model = GMM(opt)
    checkpoint = torch.load(opt.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)  # allow mismatch-safe loading
    model.eval()

    # Run inference
    for i, batch in enumerate(data_loader):
        if batch is None:
            continue
        cloth = batch["cloth"]
        person = batch["person"]
        cloth_name = batch["cloth_name"]
        person_name = batch["person_name"]

        # Forward pass (replace with your GMM inference code)
        output = model(person, cloth)

        # Save output (example, replace with your saving code)
        for j in range(cloth.size(0)):
            out_path = os.path.join(opt.result_dir, f"{i*opt.batch_size+j:06d}.png")
            # Convert tensor to PIL and save
            from torchvision.utils import save_image
            save_image(output[j], out_path)

        print(f"[Step {i}] Saved {i*opt.batch_size+j:06d}.png")

if __name__ == "__main__":
    main()
