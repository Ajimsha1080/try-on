import os
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from cp_dataset import CPDataset
from networks import GMM, UnetGenerator


# ----------------------------
# Helper functions
# ----------------------------

def save_images(cloth, image, result, save_dir, step):
    """Save cloth, image, and result side by side."""
    os.makedirs(save_dir, exist_ok=True)
    # Concatenate along width
    combined = torch.cat([cloth, image, result], 3)
    vutils.save_image(
        combined,
        os.path.join(save_dir, f"step_{step:06d}.png"),
        nrow=1,
        normalize=True
    )

def tensorboard_visualize(*args, **kwargs):
    """Dummy function (optional: expand to real TensorBoard logging)."""
    pass


# ----------------------------
# Main testing
# ----------------------------

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="GMM", help="Name of the experiment")
    parser.add_argument("--gpu_ids", default="0", help="GPU ids")
    parser.add_argument("--workers", type=int, default=1, help="DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--dataroot", default="data", help="Dataset root")
    parser.add_argument("--datamode", default="test", help="train/test/val")
    parser.add_argument("--stage", default="GMM", help="Stage: GMM or TOM")
    parser.add_argument("--data_list", default="test_pairs.txt", help="Pairs file")
    parser.add_argument("--fine_width", type=int, default=192, help="Cloth width")
    parser.add_argument("--fine_height", type=int, default=256, help="Cloth height")
    parser.add_argument("--radius", type=int, default=5, help="For TPS transformation")
    parser.add_argument("--grid_size", type=int, default=5, help="For TPS grid")
    parser.add_argument("--tensorboard_dir", default="tensorboard", help="TensorBoard dir")
    parser.add_argument("--result_dir", default="result", help="Results dir")
    parser.add_argument("--checkpoint", default="checkpoints/GMM/gmm_final.pth", help="Model checkpoint")
    parser.add_argument("--display_count", type=int, default=1, help="How often to save images")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset")
    return parser.parse_args()


def test_gmm(opt, data_loader, model):
    model.eval()
    with torch.no_grad():
        for step, inputs in enumerate(data_loader):
            cloth = inputs["cloth"].cuda()
            image = inputs["image"].cuda()

            # Forward pass
            result = model(cloth, image)

            # Save images
            if step % opt.display_count == 0:
                save_images(cloth, image, result, opt.result_dir, step)


def main():
    opt = get_opt()
    print(opt)
    print(f"Start to test stage: {opt.stage}, named: {opt.name}!")

    dataset = CPDataset(opt)
    data_loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=opt.shuffle,
        num_workers=opt.workers
    )

    if opt.stage == "GMM":
        model = GMM(opt).cuda()
    else:
        model = UnetGenerator(3, 3, 6, ngf=64, norm_layer=torch.nn.InstanceNorm2d).cuda()

    if os.path.exists(opt.checkpoint):
        model.load_state_dict(torch.load(opt.checkpoint, map_location="cuda"))
        print(f"Loaded checkpoint {opt.checkpoint}")
    else:
        print(f"Warning: checkpoint {opt.checkpoint} not found!")

    if opt.stage == "GMM":
        test_gmm(opt, data_loader, model)
    else:
        print("TOM stage not implemented yet.")


if __name__ == "__main__":
    main()
