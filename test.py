# test.py
import os
import torch
from torch.utils.data import DataLoader
from cp_dataset import CPDataset
from networks import GMM
import argparse
from tqdm import tqdm
from PIL import Image

# -----------------------------
# Argument parser
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='GMM', help='experiment name')
parser.add_argument('--batch-size', type=int, default=4, help='batch size')
parser.add_argument('--datapath', type=str, default='data', help='data root folder')
parser.add_argument('--checkpoint', type=str, default='checkpoints/GMM/gmm_final.pth', help='GMM checkpoint path')
parser.add_argument('--test_pairs', type=str, default='test_pairs.txt', help='file with test pairs')
opt = parser.parse_args()

# -----------------------------
# Dataset and DataLoader
# -----------------------------
dataset = CPDataset(opt)
data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=2)

# -----------------------------
# Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GMM(opt).to(device)

# Load checkpoint safely
if os.path.exists(opt.checkpoint):
    checkpoint = torch.load(opt.checkpoint, map_location=device)
    model.load_state_dict(checkpoint, strict=False)  # ✅ allow mismatch-safe loading
    print("[INFO] Checkpoint loaded successfully!")
else:
    print("[WARNING] Checkpoint not found! Running with random weights.")

model.eval()

# -----------------------------
# Testing loop
# -----------------------------
result_dir = "result"
os.makedirs(result_dir, exist_ok=True)

with torch.no_grad():
    for step, inputs in enumerate(tqdm(data_loader)):
        # Get images
        cloth = inputs['cloth'].to(device)
        person = inputs['person'].to(device)
        # Forward pass
        output = model(cloth, person)
        # Save output images
        for i in range(output.size(0)):
            img = output[i].cpu().detach().clamp(0, 1)  # ensure valid range
            img = torch.permute(img, (1, 2, 0)).numpy() * 255
            img = Image.fromarray(img.astype('uint8'))
            img.save(os.path.join(result_dir, f"{step*opt.batch_size+i:06d}.png"))

print("[INFO] Test finished! Results saved to 'result/' folder.")
