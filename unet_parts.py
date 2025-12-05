import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)
#down scaling with max pooling
class DownSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2 , stride=2)
    def forward(self, x):
        conv_out = self.conv(x)
        pooled_out = self.pool(conv_out)
        return conv_out, pooled_out
class UpSample(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=1, bilinear=True):
        super().__init__()

        self.in_conv = DoubleConv(in_ch, 64)
        self.down1 = DownSample(64, 128)
        self.down2 = DownSample(128, 256)
        self.down3 = DownSample(256, 512)

        # bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # upsampling path
        self.up3 = UpSample(1024, 512, bilinear)
        self.up2 = UpSample(512, 256, bilinear)
        self.up1 = UpSample(256, 128, bilinear)
        self.up0 = UpSample(128, 64, bilinear)

        self.out_conv = OutConv(64, num_classes)

    def forward(self, x):
        # down
        x0 = self.in_conv(x)           # (B, 64, H, W)
        x1, p1 = self.down1(x0)        # (B,128,H/2,W/2)
        x2, p2 = self.down2(p1)        # (B,256,H/4,W/4)
        x3, p3 = self.down3(p2)        # (B,512,H/8,W/8)

        # bottleneck
        xb = self.bottleneck(p3)

        # up path
        x = self.up3(xb, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)

        return self.out_conv(x)

