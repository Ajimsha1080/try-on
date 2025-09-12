import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """
    Feature extractor network for images (fixed to use 3-channel RGB input instead of 22).
    """

    def __init__(self, in_channels=3):  # fixed from 22 → 3
        super(FeatureExtractor, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class GMM(nn.Module):
    """
    Geometric Matching Module (simplified for RGB input).
    """

    def __init__(self):
        super(GMM, self).__init__()
        # Extractor for clothing
        self.extractionA = FeatureExtractor(in_channels=3)
        # Extractor for person image
        self.extractionB = FeatureExtractor(in_channels=3)

        # Matching layers
        self.matching = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1),  # output: flow field
        )

    def forward(self, inputA, inputB):
        featureA = self.extractionA(inputA)
        featureB = self.extractionB(inputB)

        # Concatenate features
        matching_input = torch.cat([featureA, featureB], dim=1)

        # Predict flow
        flow = self.matching(matching_input)

        return flow
