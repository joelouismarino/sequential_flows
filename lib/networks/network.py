import torch
import torch.nn as nn


class Network(nn.Module):
    """
    Base class for a neural network.
    """
    def __init__(self, n_layers, inputs=None, connectivity='sequential'):
        super(Network, self).__init__()
        self.inputs = inputs
        self.n_out = None
        self.layers = nn.ModuleList([None for _ in range(n_layers)])
        self.connectivity = connectivity
        if self.connectivity == 'highway':
            self.gates = nn.ModuleList([None for _ in range(n_layers)])

    def forward(self, input=None, **kwargs):
        if input is None:
            input = torch.cat([kwargs[k] for k in self.inputs], dim=1)
        out = input
        for ind, layer in enumerate(self.layers):
            if 'Conv' not in type(layer).__name__:
                out = out.view(out.size(0), -1)

            if 'TransposedConv' in type(layer).__name__ and len(out.size()) == 2:
                out = out.unsqueeze(2).unsqueeze(3)

            if self.connectivity == 'sequential':
                out = layer(out)

            elif self.connectivity == 'residual':
                new_out = layer(out)
                if ind == 0:
                    out = new_out
                else:
                    out = new_out + out
            elif self.connectivity == 'highway':
                new_out = layer(out)
                if ind > 0:
                    gate_out = self.gates[ind](out)
                    out = gate_out * out + (1. - gate_out) * new_out
                else:
                    out = new_out
            elif self.connectivity == 'concat':
                out = torch.cat([layer(out), out], dim=1)
            elif self.connectivity == 'concat_input':
                out = torch.cat([layer(out), input], dim=1)

            # if ind == 0 and 'trans_conv_init_size' in dir(self):
            #     out = out.view([out.size(0)] + self.trans_conv_init_size)

        return out

    def step(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass
