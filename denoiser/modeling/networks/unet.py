""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, activation, mid_channels=None, use_bn=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if use_bn:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                activation,
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                activation
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                activation,
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                activation
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, activation, use_bn=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, activation, use_bn=use_bn)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, activation, bilinear=False, use_bn=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, activation, in_channels // 2, use_bn=use_bn)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, activation, use_bn=use_bn)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, activation_type="leaky_relu", bilinear=False,
                 residual=False, extra_layer=False, merge_features=False, num_sim=32, use_bn=False):
        super(UNet, self).__init__()
        if activation_type == "leaky_relu":
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_type == "relu":
            activation = nn.ReLU(inplace=True)
        else:
            raise TypeError

        self.merge_features = merge_features
        self.num_sim = num_sim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 128, activation, use_bn=use_bn)

        self.down1 = Down(128, 256, activation, use_bn=use_bn)
        self.down2 = Down(256, 512, activation, use_bn=use_bn)
        self.drop2 = nn.Dropout2d(0.5)
        self.down3 = Down(512, 1024, activation, use_bn=use_bn)
        self.drop3 = nn.Dropout2d(0.5)

        self.up1 = Up(1024, 512, activation, bilinear, use_bn=use_bn)
        self.up2 = Up(512, 256, activation, bilinear, use_bn=use_bn)
        self.up3 = Up(256, 128, activation, bilinear, use_bn=use_bn)


        self.outc = OutConv(128, n_classes)
        self.residual = residual
        if extra_layer:
            self.extra_layer = DoubleConv(48*2, 48*2, activation, use_bn=use_bn)
        else:
            self.extra_layer = None

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)


        if self.merge_features:
            x = x.view([-1, self.num_sim, x.shape[1], x.shape[2], x.shape[3]])
            x = x.mean(dim=1)

        # print(self.merge_features)

        if self.extra_layer is not None:
            x = self.extra_layer(x)

        x = self.outc(x)

        if self.residual:
            x = input + x
        return x


class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, residual=False, activation_type="relu", use_bn=True):
        super(UNet2, self).__init__()

        if activation_type == "leaky_relu":
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_type == "relu":
            activation = nn.ReLU(inplace=True)
        else:
            raise TypeError

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 96, activation, use_bn=use_bn)
        self.down1 = Down(96, 96*2, activation, use_bn=use_bn)
        self.down2 = Down(96*2, 96*4, activation, use_bn=use_bn)
        self.down3 = Down(96*4, 96*8, activation, use_bn=use_bn)
        
        self.up1 = Up(96*8, 96*4, activation, use_bn=use_bn)
        self.up2 = Up(96*4, 96*2, activation, use_bn=use_bn)
        self.up3 = Up(96*2, 96*1, activation, use_bn=use_bn)
        self.outc = OutConv(96, n_classes)
        self.residual = residual

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        if self.residual:
            x = input + x
        return x