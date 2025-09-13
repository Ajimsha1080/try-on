import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CPDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        # Read test pairs
        with open(opt.test_pairs, "r") as f:
            self.pairs = [line.strip().split() for line in f.readlines()]

        self.transform = transforms.Compose([
            transforms.Resize((opt.fine_height, opt.fine_width)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        try:
            c_name, p_name = self.pairs[idx]

            c_path = os.path.join(self.opt.dataroot, self.opt.datamode, "cloth", c_name)
            p_path = os.path.join(self.opt.dataroot, self.opt.datamode, "person", p_name)

            cloth = Image.open(c_path).convert("RGB")
            person = Image.open(p_path).convert("RGB")

            cloth = self.transform(cloth)
            person = self.transform(person)

            return {"cloth": cloth, "person": person, "cloth_name": c_name, "person_name": p_name}

        except FileNotFoundError:
            print(f"[Warning] Skipping missing files for pair: {c_name}, {p_name}")
            return None  # Skip missing files
