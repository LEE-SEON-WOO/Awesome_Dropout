'''
    L. Wan, M. Zeiler, S. Zhang, Y. LeCun, and R. Fergus, Regularization of neural networks using dropconnect
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout

def conv_bn_act(in_, out_, kernel_size,
                stride=1, groups=1, bias=True,
                eps=1e-3, momentum=0.01):
    return nn.Sequential(
        SamePadConv2d(in_, out_, kernel_size, stride, groups=groups, bias=bias),
        nn.BatchNorm2d(out_, eps, momentum),
        Swish()
    )


class SamePadConv2d(nn.Conv2d):
    """
    Conv with TF padding='same'
    https://github.com/pytorch/pytorch/issues/3867#issuecomment-349279036
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_mode="zeros"):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias, padding_mode)
    def get_pad_odd(self, in_, weight, stride, dilation):
        effective_filter_size_rows = (weight - 1) * dilation + 1
        out_rows = (in_ + stride - 1) // stride
        padding_needed = max(0, (out_rows - 1) * stride + effective_filter_size_rows - in_)
        padding_rows = max(0, (out_rows - 1) * stride + (weight - 1) * dilation + 1 - in_)
        rows_odd = (padding_rows % 2 != 0)
        return padding_rows, rows_odd

    def forward(self, x):
        padding_rows, rows_odd = self.get_pad_odd(x.shape[2], self.weight.shape[2], self.stride[0], self.dilation[0])
        padding_cols, cols_odd = self.get_pad_odd(x.shape[3], self.weight.shape[3], self.stride[1], self.dilation[1])

        if rows_odd or cols_odd:
            x = F.pad(x, [0, int(cols_odd), 0, int(rows_odd)])

        return F.conv2d(x, self.weight, self.bias, self.stride,
                        padding=(padding_rows // 2, padding_cols // 2),
                        dilation=self.dilation, groups=self.groups)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class SEModule(nn.Module):
    def __init__(self, in_, squeeze_ch):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_, squeeze_ch, kernel_size=1, stride=1, padding=0, bias=True),
            Swish(),
            nn.Conv2d(squeeze_ch, in_, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        return x * torch.sigmoid(self.se(x))
    
class DropConnect2(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = 1.0 - ratio

    def forward(self, x):
        if not self.training:
            return x

        random_tensor = self.ratio
        random_tensor += torch.rand([x.shape[0], 1, 1, 1], dtype=torch.float, device=x.device)
        random_tensor.requires_grad_(False)
        return x / self.ratio * random_tensor.floor()

class DropConnect(nn.Module):
    '''
    https://discuss.pytorch.org/t/dropconnect-implementation/70921
    '''
    def __init__(self, module: nn.Module, p: float = 0.1, inplace = True):
        super().__init__()
        self.module = module
        self.p = p
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.module.training:
            with torch.no_grad():
                original_params = []
                for param in self.module.parameters():
                    original_params.append(param.clone())
                    param.copy_(F.dropout(param, p = self.p, inplace = self.inplace) * (1 - self.p))
        out = self.module(x)
        if self.module.training:
            with torch.no_grad():
                for original_param, param in zip(original_params, self.module.parameters()):
                    param.copy_(original_param)
        return out

if __name__ == '__main__':
    conv = nn.Conv2d(3, 3, 1, 1)
    x = torch.ones(1, 3, 4, 4)
    y = conv(x)
    print(y.shape)
    dropconnect_conv = DropConnect(conv, 0.5)
    print(dropconnect_conv(x))
    dc = DropConnect2(ratio=0.5)
    print(dc(x))