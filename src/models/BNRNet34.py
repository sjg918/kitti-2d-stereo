import torch
from torch import nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(nn.functional.softplus(x)))
        return x


class Conv(nn.Module):
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


class ResBlock(nn.Module):
    def __init__(self, in_channels, neck_channels, activation):
        super().__init__()
        self.conv1 = Conv(in_channels, neck_channels, 3, 1, activation)
        self.conv2 = Conv(neck_channels, in_channels, 3, 1, activation)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class ResBlock_DilationBottleNeck(nn.Module):
    def __init__(self, in_channels, activation):
        super().__init__()
        in_ch = in_channels
        self.conv1 = Conv(in_ch, in_ch, 3, 1, activation)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(in_ch),
            Mish(),
        )
        self.conv3 = Conv(in_ch*2, in_ch, 3, 1, activation)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(torch.cat((x1, x2), dim=1))
        return x3 + x


class DownSample1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            Conv(3, 64, 7, 2, 'mish'),
            ResBlock(64, 128, activation='mish'),
        )

    def forward(self, input):
        return self.conv1(input)

class DownSample2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            Conv(64, 64, 5, 2, 'mish'),
            ResBlock(64, 128, activation='mish'),
            ResBlock(64, 128, activation='mish'),
            ResBlock(64, 128, activation='mish'),
            ResBlock(64, 128, activation='mish'),
            )

    def forward(self, input):
        return self.conv1(input)

class DownSample3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            Conv(64, 128, 5, 2, 'mish'),
            ResBlock_DilationBottleNeck(128, activation='mish'),
            ResBlock_DilationBottleNeck(128, activation='mish'),
            ResBlock_DilationBottleNeck(128, activation='mish'),
            ResBlock_DilationBottleNeck(128, activation='mish'),
            ResBlock_DilationBottleNeck(128, activation='mish'),
            ResBlock_DilationBottleNeck(128, activation='mish'),
            ResBlock_DilationBottleNeck(128, activation='mish'),
            )

    def forward(self, input):
        return self.conv1(input)

class DownSample4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            Conv(128, 256, 3, 2, 'mish'),
            ResBlock(256, 512, activation='mish'),
            )

    def forward(self, input):
        return self.conv1(input)

class DownSample5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            Conv(256, 512, 3, 2, 'mish'),
            ResBlock(512, 1024, activation='mish'),
            )

    def forward(self, input):
        return self.conv1(input)

class BNRNet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DownSample1()
        self.down2 = DownSample2()
        self.down3 = DownSample3()
        self.down4 = DownSample4()
        self.down5 = DownSample5()

    def forward(self, input, mode='left'):
        if mode == 'left':
            out1 = self.down1(input)
            out2 = self.down2(out1)
            out3 = self.down3(out2)
            out4 = self.down4(out3)
            out5 = self.down5(out4)
            return out1, out2, out3, out4, out5
        elif mode == 'right':
            out1 = self.down1(input)
            out2 = self.down2(out1)
            out3 = self.down3(out2)
            return out3



class FC100Layer(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = lambda x: x.view(x.size(0), -1)
        self.fc = nn.Linear(input_size, 100)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, input):
        out = self.avgpool(input)
        out = self.flatten(out)
        out = self.fc(out)
        return out

class mycrossentropyloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, target):
        a = self.softmax(input)
        #loss = -1.0 * torch.sum(target * torch.log(a)) / input.shape[0]
        loss = -1.0 * torch.mean(torch.sum(target * torch.log(a), dim=1))
        return loss

if __name__ == '__main__':
    import time

    model = BNRNet34().cuda()
    x = torch.randn(1, 3, 256, 768).cuda()
    
    # cold start
    backout = model(x)
    #print(backout.shape)
    print(backout[0].shape)
    print(backout[1].shape)
    print(backout[2].shape)
    print(backout[3].shape)
    print(backout[4].shape)

    # test
    timelist = []
    with torch.no_grad():
        for i in range(250):
            torch.cuda.synchronize()
            t = time.time()            
            backout = model(x)

            torch.cuda.synchronize()
            timelist.append(time.time() - t)
        print(sum(timelist) / len(timelist))
