import torch
from .conv_lstm import ConvLSTMCell

class MultiplicativeBlock(torch.nn.Module):
    def __init__(self, c_in, c_out):
        super(MultiplicativeBlock, self).__init__()

        self.g1 = torch.nn.Conv2d(c_in, c_out, 3, 1, 1)
        self.g2 = torch.nn.Conv2d(c_in, c_out, 3, 1, 1)
        self.g3 = torch.nn.Conv2d(c_in, c_out, 3, 1, 1)
        self.u = torch.nn.Conv2d(c_in, c_out, 3, 1, 1)

    def forward(self, x):
        x1 = torch.sigmoid(self.g1(x))
        x2 = torch.sigmoid(self.g2(x))
        x3 = torch.sigmoid(self.g3(x))
        x4 = torch.tanh(self.u(x))

        out = x1 * torch.tanh(x2 * x + x3 * x4)

        return out


class ResidueMultiplicativeBlock(torch.nn.Module):
    def __init__(self, c_in):
        super(ResidueMultiplicativeBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(c_in, c_in//2, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(c_in//2, c_in, 3, 1, 1)

        self.mb1 = MultiplicativeBlock(c_in//2, c_in//2)
        self.mb2 = MultiplicativeBlock(c_in//2, c_in//2)

    def forward(self, x):
        h1 = self.conv1(x)
        h2 = self.mb1(h1)
        h3 = self.mb2(h2)
        h4 = self.conv2(h3)

        out = x + h4

        return out

class ResidueMultiplicativeNetwork(torch.nn.Module):
    def __init__(self, c_in, constant_scale, c_rmb=64, encoder_depth=2, decoder_depth=2, lstm_hidden=64):
        super(ResidueMultiplicativeNetwork, self).__init__()
        c_rmb = c_rmb
        self.init_conv = torch.nn.Conv2d(c_in, c_rmb, 3, 1, 1)
        self.encoder = torch.nn.Sequential(*[ResidueMultiplicativeBlock(c_rmb) for _ in range(encoder_depth)])
        self.decoder = torch.nn.Sequential(*[ResidueMultiplicativeBlock(c_rmb) for _ in range(decoder_depth)])
        self.init_decoder = torch.nn.Conv2d(lstm_hidden, c_rmb, 3, 1, 1)

        self.conv_lstm = ConvLSTMCell([64, 64], c_rmb, lstm_hidden, [3, 3], bias=True)
        self.init_lstm_hidden = self.conv_lstm.init_hidden(1)
        self.cur_lstm_hidden = None

        self.shift_net = torch.nn.Conv2d(c_rmb, c_in, 3, 1, 1)
        self.scale_net = torch.nn.Conv2d(c_rmb, c_in, 3, 1, 1)

        self._is_first_step = True
        self.constant_scale = constant_scale

    def forward(self, x):
        input_size = x.size()
        batch_size = input_size[0]

        x = self.init_conv(x)
        enc = self.encoder(x)
        if self._is_first_step:
            self.conv_lstm_hidden = self.conv_lstm(enc, [x.repeat(batch_size, 1, 1, 1) for x in self.init_lstm_hidden])
        else:
            self.conv_lstm_hidden = self.conv_lstm(enc, self.conv_lstm_hidden)

        self._is_first_step = False

        dec = self.init_decoder(self.conv_lstm_hidden[0])
        dec = self.decoder(dec)
        shift = self.shift_net(dec)

        if self.constant_scale:
            scale = torch.zeros_like(shift)
        else:
            scale = self.scale_net(dec)

        return shift, scale

    def reset(self):
        self._is_first_step = True
        self.cur_lstm_hidden = None
