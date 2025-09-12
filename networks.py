import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """
    Simple feature extractor with 2 convolutional layers.
    """
    def __init__(self, input_nc, output_nc):
        super(FeatureExtractor, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_nc, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)


class GMM(nn.Module):
    """
    Geometric Matching Module (simplified).
    Takes cloth + person images and produces aligned cloth.
    """
    def __init__(self, opt):
        super(GMM, self).__init__()

        # fixed input channels
        inputA_nc = 3   # cloth RGB
        inputB_nc = 3   # person RGB

        # feature extractors
        self.extractionA = FeatureExtractor(inputA_nc, 22)
        self.extractionB = FeatureExtractor(inputB_nc, 22)

        # correlation layer (combine features)
        self.correlation = nn.Conv2d(22 * 2, 128, kernel_size=3, stride=1, padding=1)

        # regression head (predict transformation grid)
        self.regressor = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * (opt.fine_height // 16) * (opt.fine_width // 16), 50),  # 50 params
        )

    def forward(self, cloth, person):
        featA = self.extractionA(cloth)
        featB = self.extractionB(person)

        # concat features
        corr = torch.cat([featA, featB], dim=1)
        corr = self.correlation(corr)

        # predict transformation params
        theta = self.regressor(corr)

        # here we just return theta (in practice you'd warp cloth with grid_sample)
        return theta
