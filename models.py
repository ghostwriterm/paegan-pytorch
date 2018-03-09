#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

############## autoencoder for ball sim predictions ##################

IM_CHANNELS = 1
IM_WIDTH = 28

V_SIZE = 256
BS_SIZE = 256
N_SIZE = 256
D_SIZE = 64
G_SIZE = 256

N_FILTERS = 16
EP_LEN = 100


class Encoder(nn.Module):
    def __init__(self, v_size=V_SIZE):
        super(Encoder, self).__init__()
        self.conv_seq = nn.Sequential(
            # out_size = (in_size - kernel_size + 2*padding)/stride + 1
            # size: (channels, 28, 28)
            nn.Conv2d(IM_CHANNELS, N_FILTERS, kernel_size=4, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            # size: (N_Filters, 16, 16)
            nn.Conv2d(N_FILTERS, N_FILTERS, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            # size: (N_FILTERS, 8, 8)
        )
        self.fc_seq = nn.Sequential(
            nn.Linear(N_FILTERS * 8 * 8, v_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h = self.conv_seq(x)
        out = self.fc_seq(h.view(h.size(0), -1))
        return out


class Decoder(nn.Module):
    def __init__(self, bs_size=BS_SIZE):
        super(Decoder, self).__init__()
        self.fc_seq = nn.Sequential(
            nn.Linear(bs_size, N_FILTERS * 8 * 8),
            nn.ReLU(inplace=True),
        )

        self.conv_seq = nn.Sequential(
            # out_size = (in_size - kernel_size + 2*padding)/stride + 1
            # size: (N_FILTERS, 8, 8)
            nn.ConvTranspose2d(N_FILTERS, N_FILTERS, kernel_size=4, stride=2, padding=2, bias=True),
            nn.ReLU(True),
            # size: (N_FILTERS, 16, 16)
            nn.ConvTranspose2d(N_FILTERS, IM_CHANNELS, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Sigmoid(),
            # size: (N_FILTERS, 28, 28)
        )

    def forward(self, x):
        h = self.fc_seq(x)
        out = self.conv_seq(h.view(h.size(0), N_FILTERS, 8, 8))
        return out


class BeliefStatePropagator(nn.Module):
    def __init__(self, v_size=V_SIZE, bs_size=BS_SIZE):
        super(BeliefStatePropagator, self).__init__()
        self.encoder = Encoder(v_size)
        self.gru = nn.GRU(v_size, bs_size, num_layers=1)

    def forward(self, x):
        ep_len = x.size(0)
        batch_size = x.size(1)

        h = self.encoder(x.view(ep_len * batch_size, x.size(2), x.size(3), x.size(4)))
        out, state_f = self.gru(h.view(ep_len, batch_size, -1))
        return out


class BeliefStateGenerator(nn.Module):
    def __init__(self, bs_size=BS_SIZE, n_size=N_SIZE, g_size=G_SIZE):
        super(BeliefStateGenerator, self).__init__()
        self.fc_seq = nn.Sequential(
            nn.Linear(bs_size + n_size, g_size),
            nn.ReLU(inplace=True),
            nn.Linear(g_size, bs_size),
            nn.Tanh(),
        )

    def forward(self, noise, bs):
        noise_joint = torch.cat([noise, bs], dim=-1)
        out = self.fc_seq(noise_joint)
        return out


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class VisualDiscriminator(nn.Module):
    def __init__(self, d=128):
        super(VisualDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        # self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        # self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 2)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)

        x = F.sigmoid(self.conv5(x))

        return x


class PAEGAN(nn.Module):
    def __init__(self):
        super(PAEGAN, self).__init__()

        self.bs_prop = BeliefStatePropagator()
        self.decoder = Decoder()
        self.D = VisualDiscriminator()
        self.G = BeliefStateGenerator()

    def forward(self):
        return None