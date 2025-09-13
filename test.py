import os
import argparse
import torch
from torch.utils.data import DataLoader
from cp_dataset import CPDataset
from networks import GMM  # or TOM if you later run TOM stage


def get_opt():
    parser = argparse.ArgumentParser()

    # Core experiment options
    parser.add_argument("--name", type=str, default="GMM", help="experiment name")
    parser.add_argument("--stage", type=str, default="GMM", help="stage: GMM or TOM")
    parser.add_argument("--workers", type=int, default=4, help="number of data loading workers")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size")

    # Dataset paths
    parser.add_argument("--dataroot", type=str, default="data", help="root path of dataset")
    parser.add_argument("--datamode", type=str, default="test", help="train or test")
    parser.add_argument("--data_list", type=str, default="test_pairs.txt", help="pairs file")

    # Checkpoint
    parser.add_argument("--checkpoint", type=str, default="checkpoints/GMM/gmm_final.pth")

    # Model hyperparameters
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)

    # GPU
    parser.add_argument("--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0,1,2")

    return parser.parse_args()


def main():
    opt = get_opt()
    print("Options:", opt)

    # Dataset + DataLoader
    dataset = CPDataset(opt)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers,
    )

    # Model
    if opt.stage == "GMM":
        model = GMM(opt)
    else:
        raise NotImplementedError(f"Stage {opt.stage} not implemented")

    if torch.cuda.is_available():
        model.cuda()

    # Load checkpoint
    if os.path.exists(opt.checkpoint):
        print(f"Loading checkpoint from {opt.checkpoint}")
        checkpoint = torch.load(opt.checkpoint, map_location="cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(checkpoint, strict=False)  # ✅ allow mismatch-safe loading
    else:
        print(f"Warning: checkpoint {opt.checkpoint} not found!")

    model.eval()

    # Inference loop
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            # Run model (simplified)
            output = model(batch)

        if i % 10 == 0:
            print(f"[{i}/{len(dataloader)}] processed")


if __name__ == "__main__":
    main()
