import torch.nn as nn  
import torch 
import numpy as np 
from torch.utils.data import Dataset
import random

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 3, 3, stride=1, padding=1),    ## 1x128x128 -> 3x128x128
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            nn.Conv2d(3, 3, 3, stride=1, padding=1),    ## 3x128x128 -> 3x128x128
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            nn.MaxPool2d(2),                            ## 3x128x128 -> 3x64x64
            nn.Conv2d(3, 32, 3, stride=1, padding=1),   ## 3x64x64 -> 32x64x64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),  ## 32x64x64 -> 32x64x64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2),                            ## 32x64x64 -> 32x32x32 
            nn.Conv2d(32, 64, 3, stride=1, padding=1), ## 32x32x32 -> 64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),## 64x32x32 -> 64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),                            ## 64x32x32 -> 64x16x16
            nn.Conv2d(64, 128, 3, stride=1, padding=1), ## 64x16x16 -> 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),## 128x16x16 -> 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),                            ## 128x16x16 -> 128x8x8
            nn.Conv2d(128, 256, 3, stride=1, padding=1), ## 128x8x8 -> 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),## 256x8x8 -> 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2),                            ## 256x8x8 -> 256x4x4
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 9, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 17, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 33, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(3, 1, 65, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x  = self.decoder(x1)
        return x1, x
        
class AE2(nn.Module):
    def __init__(self):
        super(AE2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),    ## 1x128x128 -> 16x128x128
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),    ## 16x128x128 -> 16x128x128
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2),                            ## 16x128x128 -> 16x64x64
            nn.Conv2d(16, 32, 3, stride=1, padding=1),   ## 16x64x64 -> 32x64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),  ## 32x64x64 -> 32x64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2),                            ## 32x64x64 -> 32x32x32 
            nn.Conv2d(32, 64, 3, stride=1, padding=1), ## 32x32x32 -> 64x32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),## 64x32x32 -> 64x32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2),                            ## 64x32x32 -> 64x16x16
            nn.Conv2d(64, 128, 3, stride=1, padding=1), ## 64x16x16 -> 128x16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),## 128x16x16 -> 128x16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2),                            ## 128x16x16 -> 128x8x8
            nn.Conv2d(128, 256, 3, stride=1, padding=1), ## 128x8x8 -> 256x8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),## 256x8x8 -> 256x8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2),                             ## 256x8x8 -> 256x4x4
            nn.Conv2d(256, 512, 3, stride=1, padding=1), ## 256x4x4 -> 512x4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2),                            ## 512x4x4 -> 512x2x2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=1),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(256, 128, 5, stride=1),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(128, 64, 9, stride=1),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(64, 32, 17, stride=1),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(32, 16, 33, stride=1),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(16, 1, 65, stride=1),
            nn.LeakyReLU(True),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x  = self.decoder(x1)
        return x1, x

class AE3(nn.Module):
    def __init__(self):
        super(AE3, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),    ## 1x128x128 -> 16x128x128
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),    ## 16x128x128 -> 16x128x128
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2),                            ## 16x128x128 -> 16x64x64
            nn.Conv2d(16, 32, 3, stride=1, padding=1),   ## 16x64x64 -> 32x64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),  ## 32x64x64 -> 32x64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2),                            ## 32x64x64 -> 32x32x32 
            nn.Conv2d(32, 64, 3, stride=1, padding=1), ## 32x32x32 -> 64x32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),## 64x32x32 -> 64x32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2),                            ## 64x32x32 -> 64x16x16
            nn.Conv2d(64, 128, 3, stride=1, padding=1), ## 64x16x16 -> 128x16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),## 128x16x16 -> 128x16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2),                            ## 128x16x16 -> 128x8x8
            nn.Conv2d(128, 256, 3, stride=1, padding=1), ## 128x8x8 -> 256x8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),## 256x8x8 -> 256x8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2),                             ## 256x8x8 -> 256x4x4
            nn.Conv2d(256, 512, 3, stride=1, padding=1), ## 256x4x4 -> 512x4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=1), ## 512x2x2 -> 256x4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(256, 256, 1, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(256, 128, 5, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(128, 128, 1, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(128, 64, 9, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(64, 64, 1, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(64, 32, 17, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(32, 32, 1, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(32, 16, 33, stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(16, 16, 1, stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(16, 1, 65, stride=1),
            nn.BatchNorm2d(1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x  = self.decoder(x1)
        return x1, x




class Spec_Dataset(Dataset):
    def __init__(self, X, Y):
        self.X = X 
        self.Y = Y 
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
