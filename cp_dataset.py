import os
import os.path as osp
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class CPDataset(Dataset):
    def __init__(self, opt):
        super(CPDataset, self).__init__()
        self.opt = opt
        self.data_path = opt.dataroot
        self.fine_width = opt.fine_width
        self.fine_height = opt.fine_height

        # Load pairs file (e.g., test_pairs.txt)
        pair_path = osp.join(self.data_path, opt.data_list)
        with open(pair_path, 'r') as f:
            self.pairs = [line.strip().split() for line in f.readlines()]

        # Basic transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.fine_height, self.fine_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __getitem__(self, index):
        im_name, c_name = self.pairs[index]

        # Person image
        im_path = osp.join(self.data_path, self.opt.datamode, "image", im_name)
        image = Image.open(im_path).convert("RGB")
        image = self.transform(image)

        # Cloth image
        c_path = osp.join(self.data_path, self.opt.datamode, "cloth", c_name)
        cloth = Image.open(c_path).convert("RGB")
        cloth = self.transform(cloth)

        return {
            "image": image,
            "cloth": cloth,
            "im_name": im_name,
            "c_name": c_name,
        }

    def __len__(self):
        return len(self.pairs)   # <-- This was missing
