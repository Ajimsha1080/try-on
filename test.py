import os
import torch
import argparse
from torch.utils.data import DataLoader
from networks import GMM, load_checkpoint
from cp_dataset import CPDataset
from tensorboardX import SummaryWriter

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="GMM")
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="test")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="test_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--tensorboard_dir", type=str, default="tensorboard")
    parser.add_argument("--result_dir", type=str, default="result")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--display_count", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")
    opt = parser.parse_args()
    return opt

def test_gmm(opt, test_loader, model, board):
    model.eval()
    with torch.no_grad():
        for step, inputs in enumerate(test_loader):
            im_name = inputs["im_name"]
            c_name = inputs["c_name"]
            image = inputs["image"].cuda() if torch.cuda.is_available() else inputs["image"]
            cloth = inputs["cloth"].cuda() if torch.cuda.is_available() else inputs["cloth"]
            edge = inputs["edge"].cuda() if torch.cuda.is_available() else inputs["edge"]

            # Run GMM model
            grid, theta = model(image, cloth)
            # Save results (you can expand this later)
            if step % opt.display_count == 0:
                print(f"[{step}/{len(test_loader)}] Processing {im_name} with cloth {c_name}")

def main():
    opt = get_opt()
    print(opt)

    # Tensorboard
    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))

    # Dataset & Loader
    test_dataset = CPDataset(opt)
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=opt.shuffle,
        num_workers=opt.workers
    )

    # Model
    model = GMM(opt).cuda() if torch.cuda.is_available() else GMM(opt)
    if opt.checkpoint:
        model = load_checkpoint(model, opt.checkpoint)

    # Run test
    if opt.stage == "GMM":
        test_gmm(opt, test_loader, model, board)
    else:
        print("❌ Only GMM stage implemented in this test script")

if __name__ == "__main__":
    main
