import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import your dataset + model (adjust these to match your repo)
from datasets import CPDataset, CPDataLoader
from models import GMM
from utils import save_images


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="data", help="data directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/GMM/gmm_final.pth", help="path to checkpoint")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--workers", type=int, default=4, help="number of workers")
    parser.add_argument("--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0  0,1,2  0,2. use -1 for CPU")
    parser.add_argument("--result_dir", type=str, default="results", help="save results")
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path):
    print(f"🔍 Loading checkpoint from {checkpoint_path} ...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    model_dict = model.state_dict()
    filtered_dict = {}

    for k, v in state_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            filtered_dict[k] = v
        else:
            print(f"[Skipped] {k} | checkpoint {tuple(v.shape)} != model {tuple(model_dict.get(k, torch.empty(0)).shape)}")

    # update and load
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

    print(f"✅ Loaded {len(filtered_dict)}/{len(state_dict)} layers successfully")
    return model


def main():
    opt = get_opt()
    print("Options:", opt)

    # set device
    device = torch.device("cuda" if (torch.cuda.is_available() and opt.gpu_ids != "-1") else "cpu")

    # dataset + dataloader
    dataset = CPDataset(opt.datadir)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    # model
    model = GMM(opt).to(device)
    model = load_checkpoint(model, opt.checkpoint)
    model.eval()

    os.makedirs(opt.result_dir, exist_ok=True)

    with torch.no_grad():
        for i, inputs in enumerate(dataloader):
            for key in inputs:
                inputs[key] = inputs[key].to(device)

            # forward pass
            outputs = model(inputs)

            # save results
            save_images(outputs, inputs, opt.result_dir, i)

    print("🎉 Testing complete! Results saved to", opt.result_dir)


if __name__ == "__main__":
    main()
