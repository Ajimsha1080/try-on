import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# GMM (Geometric Matching Module)
# --------------------------
class GMM(nn.Module):
    def __init__(self):
        super(GMM, self).__init__()

        # extractionA (person image feature extractor)
        self.extractionA = nn.Sequential(
            nn.Conv2d(22, 64, kernel_size=4, stride=2, padding=1),  # input channels = 22
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # extractionB (cloth image feature extractor)
        self.extractionB = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),   # input channels = 1
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Correlation layer (matches features)
        self.correlation = nn.Conv2d(128, 128, kernel_size=1)

        # Regressor to predict TPS transformation
        self.regressor = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2 * 5 * 5, kernel_size=3, padding=1)  # grid_size = 5
        )

    def forward(self, person, cloth):
        featA = self.extractionA(person)
        featB = self.extractionB(cloth)
        corr = self.correlation(featA * featB)
        theta = self.regressor(corr)
        return theta
