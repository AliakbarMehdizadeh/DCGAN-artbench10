import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()
        self.channels = channels

        self.model = nn.Sequential(
                    nn.Conv2d(self.channels, 64, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(512, 1, 4, 1, 0, bias=False),
                    nn.Sigmoid()
                )
        
    def forward(self,input):
        return self.model(input).view(-1, 1).squeeze(1)
