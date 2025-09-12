import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

# ✅ Import from cp_dataset (make sure the file is in the same folder as test.py)
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator
from utils import save_checkpoint

def test_gmm(opt, test_loader, model, board):
    model.eval()
    for step, inputs in enumerate(test_loader.data_loader):
        im = inputs['image']
        im_pose = inputs['pose']
        c = inputs['cloth']
        cm = inputs['cloth_mask']
        im_name = inputs['im_name']

        # Forward
        grid, theta = model(c, im_pose)
        warped_cloth = nn.functional.grid_sample(c, grid, padding_mode='border')

        # Save result
        save_dir = os.path.join(opt.result_dir, 'GMM')
        os.makedirs(save_dir, exist_ok=True)
        for i in range(im.size(0)):
            save_path = os.path.join(save_dir, im_name[i])
            save_image(warped_cloth[i], save_path)

        if step % opt.display_count == 0:
            print(f"Processed {step}/{len(test_loader.data_loader)}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='GMM')
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--dataroot', type=str, default='data')
    parser.add_argument('--datamode', type=str, default='test')
    parser.add_argument('--stage', type=str, default='GMM')
    parser.add_argument('--data_list', type=str, default='test_pairs.txt')
    parser.add_argument('--fine_width', type=int, default=192)
    parser
