import os
import torch
from torch.utils.data import DataLoader
from torch.nn import init
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from cp_dataset import CPDataset
from networks import GMM, UnetGenerator  # adjust imports if your model file has different names


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def test_gmm(opt, test_loader, model, board):
    model.eval()
    with torch.no_grad():
        for step, inputs in enumerate(test_loader):
            c = inputs['cloth'].cuda()
            cm = inputs['cloth_mask'].cuda()
            im = inputs['image'].cuda()
            im_pose = inputs['pose'].cuda()
            im_h = inputs['image_mask'].cuda()

            # Run GMM
            grid, theta = model(c, cm, im, im_pose)

            # Save output
            save_folder = os.path.join(opt.result_dir, opt.stage)
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, f"{step:06d}.png")
            save_image(c, save_path)

            if step % opt.display_count == 0:
                print(f"[{step}/{len(test_loader)}] Saved: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="GMM")
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dataroot", type=str, default="data")
    parser.add_argument("--datamode", type=str, default="test")
    parser.add_argument("--stage", type=str, default="GMM")
    parser.add_argument("--data_list", type=str, default="test_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--tensorboard_dir", type=str, default="tensorboard")
    parser.add_argument("--result_dir", type=str, default="result")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/GMM/gmm_final.pth")
    parser.add_argument("--display_count", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")

    opt = parser.parse_args()
    print(opt)

    os.makedirs(opt.result_dir, exist_ok=True)

    # Dataset + Loader
    test_dataset
