import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

# ✅ Use only CPDataset (no CPDataLoader in your repo)
from cp_dataset import CPDataset
from networks import GMM  # or other networks you use


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="GMM")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--datapath", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/GMM/gmm_final.pth")
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--test_pairs", type=str, default="test_pairs.txt")
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
        model.load_state_dict(checkpoint, strict=False)  # avoid crash on mismatch
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
            print(f"Saved: {result_file}")


if __name__ == "__main__":
    main()
