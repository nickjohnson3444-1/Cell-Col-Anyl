import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
         return self.conv(x)

class DownSample(nn.Module):
     def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = DoubleConv(in_ch, out_ch)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
     def forward(self, x):
         down = self.conv(x)
         p = self.pool(down)
         return down, p

class UpSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return self.conv(x)