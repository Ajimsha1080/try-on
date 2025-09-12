import os.path as osp
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class CPDataset(data.Dataset):
    def __init__(self, opt):
        super(CPDataset, self).__init__()
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode
        self.stage = opt.stage
        self.data_list = opt.data_list

        # Read pairs (image, cloth)
        im_names = []
        c_names = []
        with open(osp.join(self.root, self.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)
        self.im_names = im_names
        self.c_names = c_names

        # Transforms
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])

        # For parsing/shape (grayscale)
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width

    def __getitem__(self, index):
        im_name = self.im_names[index]
        c_name = self.c_names[index]

        # Person image
        im_path = osp.join(self.root, 'image', im_name)
        im = Image.open(im_path).convert('RGB')
        image = self.image_transform(im)

        # Cloth image
        c_path = osp.join(self.root, 'cloth', c_name)
        c = Image.open(c_path).convert('RGB')
        cloth = self.image_transform(c)

        # Cloth mask (binary)
        cm_path = osp.join(self.root, 'cloth-mask', c_name)
        cm = Image.open(cm_path).convert('L')
        cloth_mask = self.mask_transform(cm)

        # Parsing / shape original (grayscale)
        parse_shape_ori_path = osp.join(self.root, 'image-parse', im_name.replace('.jpg', '_shape.png'))
        parse_shape_ori = Image.open(parse_shape_ori_path).convert('L')
        shape_ori = self.mask_transform(parse_shape_ori)

        # Return dictionary
        return {
            'image': image,
            'cloth': cloth,
            'cloth_mask': cloth_mask,
            'shape_ori': shape_ori,
            'im_name': im_name,
            'c_name': c_name
        }

   
