import os
import torch
import argparse
from cp_dataset import CPDataset
from networks import GMM
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# ----------------------
# Argument parser
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='GMM')
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--dataroot', type=str, default='data')   # ✅ match cp_dataset.py
parser.add_argument('--datamode', type=str, default='test')   # ✅ match cp_dataset.py
parser.add_argument('--checkpoint', type=str, default='checkpoints/GMM/gmm_final.pth')
parser.add_argument('--test_pairs', type=str, default='test_pairs.txt')
parser.add_argument('--result_dir', type=str, default='result')
parser.add_argument('--workers', type=int, default=2)
opt = parser.parse_args()

# ----------------------
# Ensure result directory exists
# ----------------------
os.makedirs(opt.result_dir, exist_ok=True)

# ----------------------
# Dataset and DataLoader
# ----------------------
dataset = CPDataset(opt)
data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

# ----------------------
# Load model
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GMM(opt).to(device)

if os.path.exists(opt.checkpoint):
    print(f"Loading checkpoint from {opt.checkpoint}")
    checkpoint = torch.load(opt.checkpoint, map_location=device)
    model.load_state_dict(checkpoint, strict=False)  # allow mismatch-safe loading
else:
    print(f"[Warning] checkpoint {opt.checkpoint} not found!")

model.eval()

# ----------------------
# Run inference
# ----------------------
with torch.no_grad():
    for step, inputs in enumerate(data_loader):
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(device)

        # Forward pass
        try:
            output = model(inputs)  # adjust if your model returns a dict or tuple
        except Exception as e:
            print(f"[Error] step {step}: {e}")
            continue

        # Save results
        if isinstance(output, dict) and 'image' in output:
            result_img = output['image']
        else:
            result_img = output

        save_path = os.path.join(opt.result_dir, f"{step:06d}.png")
        save_image(result_img.cpu(), save_path)
        print(f"[Step {step}] Saved {save_path}")
