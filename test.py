import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

from cp_dataset import CPDataset
from networks import GMM  # Adjust if you also need TOM


def get_opt():
    parser = argparse.ArgumentParser()

    # Original args
    parser.add_argument("--name", type=str, default="GMM")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--datapath", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/GMM/gmm_final.pth")
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--test_pairs", type=str, default="test_pairs.txt")

    # ✅ Add back compatibility args
    parser.add_argument("--stage", type=str, default="GMM", help="Stage: GMM or TOM")
    parser.add_argument("--datamode", type=str, default="test", help="train or test")
    parser.add_argument("--data_list", type=str, default="test_pairs.txt", help="Pair list file")
    parser.add_argument("--grid_size", type=int, default=5, help="TPS grid size")

    return parser.parse_args()


def main():
    opt = get_opt()

    # ✅ Dataset + DataLoader
    dataset = CPDataset(opt)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    # ✅ Model
    model = GMM(opt)
    model = nn.DataParallel(model).cuda()

    if os.path.exists(opt.checkpoint):
        print(f"Loading checkpoint from {opt.checkpoint}")
        checkpoint = torch.load(opt.checkpoint)
        model.load_state_dict(checkpoint, strict=False)  # loose loading
    else:
        print(f"⚠️ Checkpoint not found: {opt.checkpoint}")

    # ✅ Run inference
    model.eval()
    with torch.no_grad():
        for i, inputs in enumerate(dataloader):
            cloth = inputs["cloth"].cuda()
            cloth_mask = inputs["cloth_mask"].cuda()
            im = inputs["image"].cuda()
            im_pose = inputs["pose"].cuda()
            im_mask = inputs["image_mask"].cuda()

            # forward pass
            output = model(im, im_pose, cloth, cloth_mask)

            # save result
            save_path = os.path.join("results", opt.name)
            os.makedirs(save_path, exist_ok=True)
            result_file = os.path.join(save_path, f"result_{i}.png")

            from torchvision.utils import save_image
            save_image(output, result_file)
            print(f"✅ Saved: {result_file}")


if __name__ == "__main__":
    main()
