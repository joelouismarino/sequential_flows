import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class encoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(encoder, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 64 x 64
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 32 x 32
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c5 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )

    def forward(self, x):
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        # return h5.view(-1, self.dim), [h1, h2, h3, h4]
        return h5, [h1, h2, h3, h4]

class decoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(decoder, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.LeakyReLU(0.2, inplace=True)
                )

        # state size. (nf*8) x 4 x 4
        self.upc2 = dcgan_upconv(nf * 8 * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.upc3 = dcgan_upconv(nf * 4 * 2, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.upc4 = dcgan_upconv(nf * 2 * 2, nf)
        # state size. (nf) x 32 x 32
        self.upc5_shift = nn.Sequential(
                nn.ConvTranspose2d(nf * 2, nc, 4, 2, 1),
                # state size. (nc) x 64 x 64
                )

        self.upc5_log_scale = nn.Sequential(
                nn.ConvTranspose2d(nf * 2, nc, 4, 2, 1),
                )


    def forward(self, x, skip):

        d1 = self.upc1(x)
        d2 = self.upc2(torch.cat([d1, skip[3]], 1))
        d3 = self.upc3(torch.cat([d2, skip[2]], 1))
        d4 = self.upc4(torch.cat([d3, skip[1]], 1))

        shift = self.upc5_shift(torch.cat([d4, skip[0]], 1))
        log_scale = self.upc5_log_scale(torch.cat([d4, skip[0]], 1))

        return shift, log_scale


class DCGAN_LSTM(torch.nn.Module):
    def __init__(self):
        super(DCGAN_LSTM, self).__init__()
        self.encoder = encoder(dim=128)
        self.decoder = decoder(dim=128)

        self.lstm_input_fc = torch.nn.Linear(128, 256)
        self.lstm_cell = torch.nn.LSTMCell(256, 256)
        self.lstm_output_fc = torch.nn.Linear(256, 128)

        self.reset()

    def forward(self, x):
        x, skip = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.lstm_input_fc(x)

        if self.hc is not None:
            self.hc = self.lstm_cell(x, self.hc)
        else:
            self.hc = self.lstm_cell(x)

        x = self.hc[0]
        x = F.tanh(self.lstm_output_fc(x).unsqueeze(2).unsqueeze(3))
        shift, log_scale = self.decoder(x, skip)

        return shift, log_scale

    def reset(self):
        self.hc = None

    def step(self):
        pass