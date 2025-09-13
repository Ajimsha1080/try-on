# networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtraction(nn.Module):
    def __init__(self, input_nc=22):  # Set input channels to 22
        super(FeatureExtraction, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 22, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(22*48*64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6)  # affine transformation
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

class GMM(nn.Module):
    def __init__(self, opt=None):
        super(GMM, self).__init__()
        self.extractionA = FeatureExtraction(input_nc=22)
        self.extractionB = FeatureExtraction(input_nc=22)
        self.regressor = Regressor()

    def forward(self, cloth, person):
        featA = self.extractionA(cloth)
        featB = self.extractionB(person)
        combined = featA + featB
        theta = self.regressor(combined)
        # For simplicity, return combined as output (adjust based on your original GMM forward)
        return combined
