import os
import argparse
import torch
from torch.utils.data import DataLoader

from cp_dataset import CPDataset
from networks import GMM, UnetGenerator
from utils import save_images, tensorboard_visualize

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="GMM", help="name of the experiment")
    parser.add_argument("--gpu_ids", default="0", help="gpu ids: e.g. 0  0,1,2  0,2")
    parser.add_argument("--workers", type=int, default=1, help="number of data loading workers")
    parser.add_argument("--batch_size", type=int, default=4, help="input batch size")
    parser.add_argument("--dataroot", default="data", help="root directory of dataset")
    parser.add_argument("--datamode", default="test", help="train or test")
    parser.add_argument("--stage", default="GMM", help="stage: GMM or TOM")
    parser.add_argument("--data_list", default="test_pairs.txt", help="data list file")
    parser.add_argument("--fine_width", type=int, default=192, help="resized image width")
    parser.add_argument("--fine_height", type=int, default=256, help="resized image height")
    parser.add_argument("--radius", type=int, default=5, help="radius for GMM")
    parser.add_argument("--grid_size", type=int, default=5, help="grid size for GMM")
    parser.add_argument("--tensorboard_dir", default="tensorboard", help="save tensorboard logs here")
    parser.add_argument("--result_dir", default="result", help="save results here")
    parser.add_argument("--checkpoint", default="", help="path to checkpoint")
    parser.add_argument("--display_count", type=int, default=1, help="frequency of showing training results on screen")
    parser.add_argument("--shuffle", action="store_true", help="whether to shuffle the dataset")
    return parser.parse_args()

def test_gmm(opt, test_loader, model):
    model.eval()
    os.makedirs(opt.result_dir, exist_ok=True)
    with torch.no_grad():
        for step, inputs in enumerate(test_loader):
            c = inputs["cloth"].cuda()
            cm = inputs["cloth_mask"].cuda()
            im = inputs["image"].cuda()
            im_pose = inputs["pose"].cuda()

            # Forward pass through the model
            warped_cloth, warped_grid = model(c, cm, im_pose)

            # Save results
            save_images(c, im, warped_cloth, opt.result_dir, step)

            if step % opt.display_count == 0:
                print(f"[Step {step}] Saved results")

def test_tom(opt, test_loader, model):
    model.eval()
    os.makedirs(opt.result_dir, exist_ok=True)
    with torch.no_grad():
        for step, inputs in enumerate(test_loader):
            c = inputs["cloth"].cuda()
            cm = inputs["cloth_mask"].cuda()
            im = inputs["image"].cuda()
            im_pose = inputs["pose"].cuda()
            im_g = inputs["image_masked"].cuda()
            im_c = inputs["cloth_masked"].cuda()

            output = model(torch.cat([im_c, im_pose], 1))
            p_rendered, m_composite = torch.split(output, [3, 1], 1)
            p_rendered = torch.tanh(p_rendered)
            m_composite = torch.sigmoid(m_composite)
            im_tryon = c * m_composite + p_rendered * (1 - m_composite)

            save_images(c, im, im_tryon, opt.result_dir, step)

            if step % opt.display_count == 0:
                print(f"[Step {step}] Saved results")

def main():
    opt = get_opt()
    print(f"Start to test stage: {opt.stage}, named: {opt.name}!")

    dataset = CPDataset(opt)
    test_loader = DataLoader(dataset, batch_size=opt.batch_size,
                             shuffle=opt.shuffle, num_workers=opt.workers)

    if opt.stage == "GMM":
        model = GMM(opt.fine_height, opt.fine_width, opt.radius, opt.grid_size).cuda()
    elif opt.stage == "TOM":
        model = UnetGenerator(22, 4, 6, ngf=64, norm_layer=torch.nn.InstanceNorm2d).cuda()
    else:
        raise ValueError(f"Unknown stage: {opt.stage}")

    if opt.checkpoint and os.path.exists(opt.checkpoint):
        print(f"Loading checkpoint from {opt.checkpoint}")
        model.load_state_dict(torch.load(opt.checkpoint))
    else:
        print("Warning: No checkpoint found, running with random weights!")

    if opt.stage == "GMM":
        test_gmm(opt, test_loader, model)
    else:
        test_tom(opt, test_loader, model)

if __name__ == "__main__":
    main()
