import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class CPDataset(Dataset):
    def __init__(self, opt):
        super(CPDataset, self).__init__()
        self.opt = opt

        # Read pairs from list file
        with open(opt.data_list, "r") as f:
            pairs = [line.strip().split() for line in f.readlines()]

        # Validate pairs (skip missing files)
        self.pairs = []
        for c_name, i_name in pairs:
            c_path = os.path.join(opt.dataroot, opt.datamode, "cloth", c_name)
            i_path = os.path.join(opt.dataroot, opt.datamode, "image", i_name)
            a_path = os.path.join(opt.dataroot, opt.datamode, "agnostic-v3.2", i_name)

            if not (os.path.isfile(c_path) and os.path.isfile(i_path) and os.path.isfile(a_path)):
                print(f"[Warning] Skipping missing files for pair: {c_name}, {i_name}")
                continue

            self.pairs.append((c_name, i_name))

        # Define transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((opt.fine_height, opt.fine_width)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __getitem__(self, index):
        c_name, i_name = self.pairs[index]

        c_path = os.path.join(self.opt.dataroot, self.opt.datamode, "cloth", c_name)
        i_path = os.path.join(self.opt.dataroot, self.opt.datamode, "image", i_name)
        a_path = os.path.join(self.opt.dataroot, self.opt.datamode, "agnostic-v3.2", i_name)

        # Load images (we already filtered missing files)
        cloth = Image.open(c_path).convert("RGB")
        image = Image.open(i_path).convert("RGB")
        agnostic = Image.open(a_path).convert("RGB")

        # Apply transforms
        cloth = self.transform(cloth)
        image = self.transform(image)
        agnostic = self.transform(agnostic)

        return {
            "cloth": cloth,
            "image": image,
            "agnostic": agnostic,
            "c_name": c_name,
            "i_name": i_name,
        }

    def __len__(self):
        return len(self.pairs)


def get_loader(opt, shuffle=False):
    dataset = CPDataset(opt)
    data_loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        num_workers=opt.workers,
    )
    return data_loader
