import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, stride,
                 conv='conv', padding='mirror', norm='in', activation='relu',
                 sn=False):
        super(ConvBlock, self).__init__()
        if padding == 'mirror':
            self.padding = nn.ReflectionPad2d(kernel_size // 2)
        elif padding == 'none':
            self.padding = None
        else:
            self.padding = nn.ReflectionPad2d(padding)

        if conv == 'conv':
            self.conv = nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride)
        elif conv == 'trans':
            self.conv = nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size, stride=stride)

        if norm == 'in':
            self.norm = nn.InstanceNorm2d(out_size, affine=True)
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(out_size)
        elif norm == 'none':
            self.norm = None

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'none':
            self.activation = None

        if sn:
            self.conv = spectral_norm(self.conv)

    def forward(self, x):
        if self.padding:
            out = self.padding(x)
        else:
            out = x
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_size, kernel_size, stride,
                 conv='conv', padding='mirror', norm='in', activation='relu',
                 sn=False):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(in_size, in_size, kernel_size, stride, conv, padding, norm, activation, sn),
            ConvBlock(in_size, in_size, kernel_size, stride, conv, padding, norm, activation, sn)
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_size=3, activation='leakyrelu'):
        super(Encoder, self).__init__()
        self.conv_1 = ConvBlock(in_size, 32, kernel_size=9, stride=1,
                                activation=activation, sn=True)
        self.conv_2 = ConvBlock(32, 64, kernel_size=3, stride=2,
                                activation=activation, sn=True)
        self.conv_3 = ConvBlock(64, 128, kernel_size=3, stride=2,
                                activation=activation, sn=True)
        self.res_block = nn.Sequential(
            ResBlock(128, kernel_size=3, stride=1, activation=activation, sn=True),
            ResBlock(128, kernel_size=3, stride=1, activation=activation, sn=True),
            ResBlock(128, kernel_size=3, stride=1, activation=activation, sn=True),
            ResBlock(128, kernel_size=3, stride=1, activation=activation, sn=True)
        )

    def forward(self, x):
        out_1 = self.conv_1(x)
        out_2 = self.conv_2(out_1)
        out_3 = self.conv_3(out_2)
        out = self.res_block(out_3)
        return out, out_3, out_2


class Decoder(nn.Module):
    def __init__(self, out_size=3, activation='leakyrelu'):
        super(Decoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock(256, 64, kernel_size=3, stride=1, activation=activation, sn=True)
        )
        self.conv_2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock(128, 32, kernel_size=3, stride=1, activation=activation, sn=True)
        )
        self.conv_3 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, out_size, kernel_size=9, stride=1)
        )

    def forward(self, x, age_vec, skip_1, skip_2):
        b, c = age_vec.size()
        age_vec = age_vec.view(b, c, 1, 1)
        out = age_vec * x
        out = torch.cat((out, skip_1), 1)
        out = self.conv_1(out)
        out = torch.cat((out, skip_2), 1)
        out = self.conv_2(out)
        out = self.conv_3(out)
        return out


class AgeMod(nn.Module):
    def __init__(self):
        super(AgeMod, self).__init__()
        self.fc_mix = nn.Linear(101, 128, bias=False)

    def forward(self, x):
        b_s = x.size(0)
        z = torch.zeros(b_s, 101).type_as(x).float()
        for i in range(b_s):
            z[i, x[i]] = 1
        y = self.fc_mix(z)
        y = torch.sigmoid(y)
        return y

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = Encoder(in_size=3, activation="leakyrelu")
        self.age_mod = AgeMod()
        self.decoder = Decoder(out_size=3, activation="leakyrelu")

    def forward(self, x, age_src, age_dst):

        middle, skip_1, skip_2 = self.encoder(x)
        if age_src is not None:
            age_src = self.age_mod(age_src)
            recon_x = self.decoder(middle, age_src, skip_1, skip_2)
        else:
            recon_x = None
        age_dst = self.age_mod(age_dst)
        modif_x = self.decoder(middle, age_dst, skip_1, skip_2)

        return recon_x, modif_x, age_dst

def pretrained_generator(path, device):
    generator = Generator()

    checkpoint = torch.load(path, map_location=torch.device(device))

    generator.encoder.load_state_dict(checkpoint["enc_state_dict"])
    generator.decoder.load_state_dict(checkpoint["dec_state_dict"])
    generator.age_mod.load_state_dict(checkpoint["mlp_style_state_dict"])

    return generator


def my_generator(path, device):
    generator = Generator()
    checkpoint = torch.load(path, map_location=torch.device(device))
    generator.load_state_dict(checkpoint["gen"])
    return generator