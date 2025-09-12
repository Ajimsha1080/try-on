import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """
    Simple feature extractor with 2 convolutional layers.
    Takes an image (e.g., cloth/person RGB) and extracts features.
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
    Geometric Matching Module (simplified for test stage).
    Takes cloth + person images and predicts aligned cloth.
    """
    def __init__(self, opt):
        super(GMM, self).__init__()

        # ✅ fixed input channels
        inputA_nc = 3   # cloth RGB
        inputB_nc = 3   # person RGB

        # feature extractors
        self.extractionA = FeatureExtractor(inputA_nc, 22)
        self.ex
