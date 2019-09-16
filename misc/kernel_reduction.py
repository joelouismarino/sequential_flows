import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class Compressed_Conv2d(torch.nn.Conv2d):
    def __init__(self, basis_kernel, basis_bias, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):

        super(Compressed_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups,
                 bias, padding_mode)

        self.in_channels = in_channels
        self.out_channels = out_channels

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        self.basis_kernel = basis_kernel
        self.weights = torch.nn.Parameter(torch.eye(self.out_channels))
        self.bias = torch.nn.Parameter(basis_bias.detach())

    def forward(self, x):
        kernel = torch.matmul(self.weights, self.basis_kernel.view(self.out_channels, -1))
        kernel = kernel.view(self.basis_kernel.size())
        return F.conv2d(x, kernel, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

layer_t1 = torch.nn.Conv2d(64, 128, 4)
layer_t2 = Compressed_Conv2d(layer_t1.weight, layer_t1.bias, 64, 128, 4)

x = torch.randn(1,64,16,16)

r1 = layer_t1(x)
r2 = layer_t2(x)

print()

