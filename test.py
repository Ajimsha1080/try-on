import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

from datasets.cp_dataset import CPDataset, CPDataLoader
from networks import GMM


def get_opt():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="GMM", help="name of the experiment")
    parser.add_argument("--gpu_ids", default="0", help="gpu ids")
    parser.add_argument("--workers", type=int, default=1, help="number of data loading workers")
    parser.add_argument("--batch_size", type=int, default=4, help="input batch size")
    parser.add_argument("--dataroot", default="data", help="root directory of dataset")
    parser.add_argument("--datamode", default="test", help="train or test mode")
    parser.add_argument("--stage", default="GMM", help="stage: GMM or TOM")
    parser.add_argument("--data_list", default="test_pairs.txt", help="data list file")
    parser.add_argument("--fine_width", type=int, default=192, help="resized image width")
    parser.add_argument("--fine_height", type=int, default=256, help="resized image height")
    parser.add_argument("--radius", type=int, default=5, help="radius for mask dilation")
    parser.add_argument("--grid_size", type=int, default=5, help="grid size for TPS")
    parser.add_argument("--tensorboard_dir", default="tensorboard", help="save tensorboard logs here")
    parser.add_argument("--result_dir", default="result", help="save result images here")
    parser.add_argument("--checkpoint", default="checkpoints/GMM/gmm_final.pth", help="model checkpoint")
    parser.add_argument("--display_count", type=int, default=1, help="frequency of showing training results")
    parser.add_argument("--shuffle", action="store_true", help="shuffle input data")
    opt = parser.parse_args([])
    return opt


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    # Try flexible loading
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError as e:
        print("⚠️ Checkpoint mismatch, loading with strict=False")
        model.load_state_dict(checkpoint, strict=False)

    return model


def main():
    opt = get_opt()

    dataset = CPDataset(opt)
    data_loader = CPDataLoader(opt, dataset)

    model = GMM(opt)
    model = load_checkpoint(model, opt.checkpoint)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for step, inputs in enumerate(data_loader.data_loader):
            cloth = inputs["cloth"].cuda()
            agnostic = inputs["agnostic"].cuda()

            output = model(agnostic, cloth)

            # Save outputs
            save_dir = os.path.join(opt.result_dir, "GMM")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{step:05d}.png")
            transforms.ToPILImage()(output[0].cpu()).save(save_path)

            if step % opt.display_count == 0:
                print(f"[Step {step}] Saved {save_path}")


if __name__ == "__main__":
    main()
