import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )
        self.se = SEBlock(out_ch)

    def forward(self, x):
        x = self.mpconv(x)
        x = self.se(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)
        self.se = SEBlock(out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.se(x)
        return x

    def get_first(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        # x = self.se(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, in_ch):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(in_ch, in_ch)

    def forward(self, x):
        vector = self.pool(x).squeeze(-1).squeeze(-1)
        attention = self.linear(vector)
        attention = attention.unsqueeze(-1).unsqueeze(-1)
        return attention * x


class UNet_v11(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_v11, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.shortcut1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.shortcut2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.shortcut3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.shortcut4 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, self.shortcut4(x4))
        x = self.up2(x, self.shortcut3(x3))
        x = self.up3(x, self.shortcut2(x2))
        x = self.up4(x, self.shortcut1(x1))
        x = self.outc(x)
        return F.sigmoid(x)


class UNet_v11_test(nn.Module):
    def __init__(self, n_channels, n_classes, device1, device2, device3, device4, device5):
        super(UNet_v11_test, self).__init__()
        self.device1 = device1
        self.device2 = device2
        self.device3 = device3
        self.device4 = device4
        self.device5 = device5
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.shortcut1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.shortcut2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.shortcut3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.shortcut4 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def move(self):
        self.inc = self.inc.to(self.device1)
        self.down1 = self.down1.to(self.device1)
        self.shortcut1 = self.shortcut1.to(self.device1)
        self.down2 = self.down2.to(self.device2)
        self.shortcut2 = self.shortcut2.to(self.device2)
        self.down3 = self.down3.to(self.device3)
        self.shortcut3 = self.shortcut3.to(self.device3)
        self.down4 = self.down4.to(self.device4)
        self.shortcut4 = self.shortcut4.to(self.device4)
        self.up1 = self.up1.to(self.device1)
        self.up2 = self.up2.to(self.device2)
        self.up3 = self.up3.to(self.device3)
        self.up4 = self.up4.to(self.device4)
        self.up4.se.to(self.device5)
        self.outc = self.outc.to(self.device5)

    def forward(self, x):
        x = x.to(self.device1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = x2.to(self.device2)
        x3 = self.down2(x2)
        x3 = x3.to(self.device3)
        x4 = self.down3(x3)
        x4 = x4.to(self.device4)
        x5 = self.down4(x4)
        x5 = x5.to(self.device1)
        x = self.up1(x5, self.shortcut4(x4.to(self.device4)).to(self.device1))
        x = x.to(self.device2)
        x = self.up2(x, self.shortcut3(x3.to(self.device3)).to(self.device2))
        x = x.to(self.device3)
        x = self.up3(x, self.shortcut2(x2.to(self.device2)).to(self.device3))
        x = x.to(self.device4)
        x_final = self.shortcut1(x1.to(self.device1)).to(self.device4)
        # x = self.up4(x, x_final)
        x = self.up4.get_first(x, x_final)
        x = self.up4.se(x)
        x = x.to(self.device5)
        x = self.outc(x)
        return F.sigmoid(x)