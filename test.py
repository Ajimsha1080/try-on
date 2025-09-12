import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from cp_dataset import CPDataset
from networks import GMM


def test_gmm(opt, data_loader, model):
    model.eval()
    save_path = os.path.join(opt.result_dir, opt.stage)
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for step, inputs in enumerate(data_loader):
            # Unpack dataset batch
            im = inputs['image'].cuda()          # person image
            cm = inputs['cloth'].cuda()          # cloth image
            cm_mask = inputs['cloth_mask'].cuda() if 'cloth_mask' in inputs else None

            # Forward pass through GMM
            output = model(cm, im)

            # Save warped cloth or result
            save_name = f"{step:06d}.png"
            save_image(output, os.path.join(save_path, save_name))

            if step % opt.display_count == 0:
                print(f"[Step {step}] Saved {save_name}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="GMM")
    parser.add_argument("--gpu_ids", default="0")
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
    parser.add_argument("--shuffle", action="store_true", default=False)

    opt = parser.parse_args()
    print(opt)

    # Dataset + DataLoader
    dataset = CPDataset(opt)
    data_loader = DataLoader(dataset, batch_size=opt.batch_size,
                             shuffle=opt.shuffle, num_workers=opt.workers)

    # Model (now accepts opt)
    model = GMM(opt).cuda()

    # Load checkpoint if available
    if os.path.exists(opt.checkpoint):
        print(f"Loading checkpoint from {opt.checkpoint}")
        checkpoint = torch.load(opt.checkpoint)
        model.load_state_dict(checkpoint)
    else:
        print(f"Warning: checkpoint {opt.checkpoint} not found!")

    # Run test
    test_gmm(opt, data_loader, model)


if __name__ == "__main__":
    main()
