import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_layer(nn.Module):
    def __init__(self, nin, nout):
        super(conv_layer, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 3, 1, 1),
                nn.BatchNorm2d(nout),
                nn.ReLU(),
                )

    def forward(self, input):
        return self.main(input)


class CustomFlowNetwork(torch.nn.Module):
    def __init__(self):
        super(CustomFlowNetwork, self).__init__()

        self.init_encoder = conv_layer(1, 64)
        self.encoder = torch.nn.Sequential(*[conv_layer(64, 64) for _ in range(3)])

        self.init_decoder = conv_layer(3*64, 64)
        self.decoder = torch.nn.Sequential(*[conv_layer(64, 64) for _ in range(3)])

        self.shift_net = nn.Conv2d(64, 1, 3, 1, 1)
        self.log_scale_net = nn.Conv2d(64, 1, 3, 1, 1)


    def forward(self, x):
        batch_size, buffer_length, w, h = x.size()
        x = x.view(-1, 1, w, h)

        x = self.init_encoder(x)
        x = self.encoder(x)

        x = x.view(batch_size, -1, w, h).contiguous()
        x = self.init_decoder(x)
        x = self.decoder(x)

        shift = self.shift_net(x)
        log_scale = self.log_scale_net(x)

        return shift, log_scale

    def reset(self):
        pass

    def step(self):
        pass