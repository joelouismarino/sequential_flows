import torch
import torch.nn as nn
from .layer import Layer
from torch.autograd import Variable

class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: int
            Size of squared input tensor.
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: int
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.

        padding set at kernel_size//2 and stride set to 1 to maintain size
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size//2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())


class ConvRecurrentLayer(Layer):
    """
    An LSTM layer.
    """
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, non_linearity=None, dropout=None):
        super(ConvRecurrentLayer, self).__init__()
        self.lstm = ConvLSTMCell(input_size, input_dim, hidden_dim, kernel_size, bias=True)
        if dropout:
            self.dropout = nn.Dropout1d(dropout)

        output_shape = [hidden_dim, input_size, input_size]
        self.output_shape = output_shape
        self.initial_hidden = nn.Parameter(torch.zeros([1]+output_shape))
        self.initial_cell = nn.Parameter(torch.zeros([1]+output_shape))
        self.hidden_state = self._hidden_state = None
        self.cell_state = self._cell_state = None
        self._detach = False

        if non_linearity is None or non_linearity == 'linear':
            self.non_linearity = None
        elif non_linearity == 'relu':
            self.non_linearity = nn.ReLU()
            self.init_gain = nn.init.calculate_gain('relu')
            self.bias_init = 0.1
        elif non_linearity == 'leaky_relu':
            self.non_linearity = nn.LeakyReLU(0.2, inplace=True)
        elif non_linearity == 'elu':
            self.non_linearity = nn.ELU()
        elif non_linearity == 'selu':
            self.non_linearity = nn.SELU()
        elif non_linearity == 'tanh':
            self.non_linearity = nn.Tanh()
            self.init_gain = nn.init.calculate_gain('tanh')
        elif non_linearity == 'sigmoid':
            self.non_linearity = nn.Sigmoid()
        else:
            raise Exception('Non-linearity ' + str(non_linearity) + ' not found.')

    def forward(self, input):
        if self.hidden_state is None:
            # re-initialize the hidden state
            self.hidden_state = self.initial_hidden.repeat(input.shape[0], 1)
        if self.cell_state is None:
            # re-initialize the cell state
            self.cell_state = self.initial_cell.repeat(input.shape[0], 1)
        # detach the hidden and cell states if necessary
        hs = self.hidden_state.detach() if self._detach else self.hidden_state
        cs = self.cell_state.detach() if self._detach else self.cell_state
        # perform forward computation
        self._hidden_state, self._cell_state = self.lstm(input, (hs, cs))
        self.step()

        output = self._hidden_state
        if self.non_linearity is not None:
            output = self.non_linearity(output)

        return output

    def step(self):
        self.hidden_state = self._hidden_state
        self.cell_state = self._cell_state

    @property
    def state(self):
        return self.hidden_state.detach() if self._detach else self.hidden_state

    def reset(self, batch_size):
        self.hidden_state = self.initial_hidden.repeat([batch_size] + [1 for _ in self.output_shape])
        self.cell_state = self.initial_cell.repeat([batch_size] + [1 for _ in self.output_shape])
        self._hidden_state = self._cell_state = None
