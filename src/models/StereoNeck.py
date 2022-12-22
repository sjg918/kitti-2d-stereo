
import torch
from torch import nn
import torch.nn.functional as F


# refer : https://github.com/Tianxiaomo/pytorch-YOLOv4
# refer : https://github.com/JiaRenChang/PSMNet
# refer : https://github.com/xy-guo/GwcNet


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(nn.functional.softplus(x)))
        return x


def build_absdif_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = (refimg_fea[:, :, :, i:] - targetimg_fea[:, :, :, :-i]).abs()
        else:
            volume[:, :, i, :, :] = (refimg_fea - targetimg_fea).abs()
    return volume.contiguous()


class myConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, norm=True, bias=False):
        super().__init__()

        pad = (kernel_size - 1) // 2
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=bias))
        if norm:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == 'mish':
            self.conv.append(Mish())
        if activation == 'leaky':
            self.conv.append(nn.LeakyReLU(0.1))
        elif activation == 'linear':
            pass
        else:
            assert 'unknown activation function!'

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class myConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, norm=True, bias=False):
        super().__init__()

        pad = (kernel_size - 1) // 2
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, pad, bias=bias))
        if norm:
            self.conv.append(nn.BatchNorm3d(out_channels))
        if activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1))
        elif activation == 'mish':
            self.conv.append(Mish())
        elif activation == "linear":
            pass
        else:
            assert 'unknown activation function!'

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


def disparity_regression(x, maxdisp=192):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype).to(x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)
  
  
class HourGlass_keep(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv0 = nn.Sequential(
            myConv3D(in_channels, in_channels, 3, 1, 'leaky'),
            myConv3D(in_channels, in_channels, 3, 1, 'leaky'),
        )

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=(1,2,2), padding=(1,1,1), bias=False),
            nn.BatchNorm3d(in_channels),
            nn.LeakyReLU(0.1),
            )

        self.conv2 = myConv3D(in_channels, in_channels, 3, 1, 'leaky')

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=(1,2,2), padding=(1,1,1), bias=False),
            nn.BatchNorm3d(in_channels),
            nn.LeakyReLU(0.1),
            )

        self.conv4 = nn.Sequential(
            myConv3D(in_channels, in_channels, 3, 1, 'leaky'),
            myConv3D(in_channels, in_channels, 3, 1, 'leaky'),
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels, in_channels, 3, padding=1, output_padding=(0, 1, 1), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(in_channels))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels, in_channels, 3, padding=1, output_padding=(0, 1, 1), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(in_channels),
        )
        self.redir2 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(in_channels),
        )

    def forward(self, x):
        x = self.conv0(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.leaky_relu(self.conv5(conv4) + self.redir2(conv2), 0.1)
        conv6 = F.leaky_relu(self.conv6(conv5) + self.redir1(x), 0.1)

        return conv6, conv5, conv4


class StereoNeck_pretrain(nn.Module):
    def __init__(self, maxdisp):
        super().__init__()
        self.maxdisp = maxdisp

        self.extraconv = nn.Sequential(
            myConv2D(128, 64, 1, 1, 'leaky'),
            myConv2D(64, 64, 1, 1, 'leaky'),
            )

        self.hourglass = HourGlass_keep(64)

        self.head = nn.Sequential(
            myConv3D(64, 64, 3, 1, 'leaky'),
            myConv3D(64, 64, 3, 1, 'leaky'),
            nn.Conv3d(64, 1, 3, 1, 1)
        )


    def forward(self, left_x8, right_x8):
        left = self.extraconv(left_x8)
        right = self.extraconv(right_x8)

        B, C, H, W = left.shape
        cost = build_absdif_volume(left, right, self.maxdisp//8)

        x8, _, _ = self.hourglass(cost)
        x8 = self.head(x8)

        x = F.interpolate(x8, [self.maxdisp, H * 8, W * 8], mode='trilinear', align_corners=False).squeeze(1)
        x = F.softmax(x, dim=1)
        x = disparity_regression(x, self.maxdisp).unsqueeze(1)

        return x
      
      
class StereoNeck(nn.Module):
    def __init__(self, maxdisp):
        super().__init__()
        self.maxdisp = maxdisp

        self.extraconv = nn.Sequential(
            myConv2D(128, 64, 1, 1, 'leaky'),
            myConv2D(64, 64, 1, 1, 'leaky'),
            )

        self.hourglass = HourGlass_keep(64)

        self.head = nn.Sequential(
            myConv3D(64, 64, 3, 1, 'leaky'),
            myConv3D(64, 64, 3, 1, 'leaky'),
            nn.Conv3d(64, 1, 3, 1, 1)
        )

        self.head1 = nn.Sequential(
            myConv3D(64, 64, 3, 1, 'leaky'),
            myConv3D(64, 64, 3, 1, 'leaky'),
            myConv3D(64, 16, 3, 1, 'leaky'),
        )

    def forward(self, left_x8, right_x8, target=None):
        left = self.extraconv(left_x8)
        right = self.extraconv(right_x8)

        B, C, H, W = left.shape
        cost = build_absdif_volume(left, right, self.maxdisp//8)

        hg8, _, _ = self.hourglass(cost)
        x8 = self.head1(hg8)
        #x8 = x8.permute(0, 2, 1, 3, 4).contiguous()
        x8 = x8.view(B, 256, H, W).contiguous()

        if target is not None:
            disppred = self.head(hg8)

            disppred = F.interpolate(disppred, [self.maxdisp, H * 8, W * 8], mode='trilinear', align_corners=False).squeeze(1)
            disppred = F.softmax(disppred, dim=1)
            disppred = disparity_regression(disppred, self.maxdisp).unsqueeze(1)

            dataL = target['dataL']
            mask = target['mask']
            loss = F.smooth_l1_loss(disppred[mask], dataL[mask], reduction='mean')
            return x8, loss
        return x8
