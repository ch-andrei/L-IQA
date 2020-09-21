import numpy as np

import torch
from torch import nn


def conv_act(ch_in, ch_out, kernel_size, stride=1, padding=0):
    return nn.Conv2d(ch_in, ch_out, kernel_size, stride=stride, padding=padding), nn.GELU()


class SalCAR(nn.Module):
    def __init__(self, ch_in=64, chn_out=64):
        super(SalCAR, self).__init__()
        layers = [nn.MaxPool2d(2, stride=2), ]
        layers += [*conv_act(ch_in, ch_in, 3, stride=1, padding=1), ]
        layers += [*conv_act(ch_in, chn_out, 3, stride=1, padding=1), ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class SplitCAR(nn.Module):
    def __init__(self, ch_in=128, chn_out=256):
        super(SplitCAR, self).__init__()
        self.maxpool = nn.MaxPool2d(2, stride=2)
        layers = [*conv_act(ch_in, ch_in, 3, stride=1, padding=1),
                  *conv_act(ch_in, chn_out, 3, stride=1, padding=1)]
        self.conv = nn.Sequential(*layers)

        self.residual = nn.Conv2d(ch_in, chn_out, 1, stride=1, padding=0)

    def forward(self, x):
        x = self.maxpool(x)
        return self.conv(x) + self.residual(x)


class SalSubnet(nn.Module):
    def __init__(self):
        super(SalSubnet, self).__init__()

        conv1_2 = [*conv_act(3, 32, 3, stride=1, padding=1),
                   *conv_act(32, 32, 3, stride=1, padding=1)]
        self.conv1_2 = nn.Sequential(*conv1_2)
        self.salcar1 = SalCAR(ch_in=32, chn_out=64)
        self.salcar2 = SalCAR(ch_in=64, chn_out=128)
        self.splitcar1 = SplitCAR(ch_in=128, chn_out=256)
        self.splitcar2 = SplitCAR(ch_in=256, chn_out=512)

    def forward(self, x):
        x = self.conv1_2(x)
        x = self.salcar1(x)
        x = self.salcar2(x)
        x = self.splitcar1(x)
        x = self.splitcar2(x)
        return x


class SIQ(nn.Module):
    def __init__(self):
        super(SIQ, self).__init__()

        self.net = SalSubnet()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(512, 1),
        )

    def forward(self, in0, in1):
        f1 = self.net(in0)
        f2 = self.net(in1)
        x = f2 - f1
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_siq(use_gpu, custom_model_path):
    model = SIQ()

    device = torch.device("cuda" if use_gpu else "cpu")

    model.to(device)
    if device == torch.device("cuda"):
        # from torch.nn.parallel import DistributedDataParallel as DDP
        print("Use DataParallel if multi-GPU")
        # model = DDP(model)
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    with open(custom_model_path, "rb") as f:
        checkpoint_data = torch.load(f, map_location=device)

    model.load_state_dict(checkpoint_data["model_state_dict"])

    return model