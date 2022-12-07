# !/usr/bin/env python
# coding: utf-8
"""
S. Park and N. Kwak, Analysis on the dropout effect in convolutional neural networks

@misc{santos2020maxdropout,
    title={MaxDropout: Deep Neural Network Regularization Based on Maximum Output Values},
    author={Claudio Filipi Goncalves do Santos and Danilo Colombo and Mateus Roder and João Paulo Papa},
    year={2020},
    eprint={2007.13723},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
"""
import torch
import torch.nn as nn

class MaxDropout(nn.Module):
    def __init__(self, drop=0.3):
#         print(p)
        super(MaxDropout, self).__init__()
        if drop < 0 or drop > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.drop = 1 - drop

    def forward(self, x):
        if not self.training:
            return x

        up = x - x.min()
        divisor =  (x.max() - x.min())
        x_copy = torch.div(up,divisor)
        if x.is_cuda:
            x_copy = x_copy.cuda()

        mask = (x_copy > (self.drop))
        x = x.masked_fill(mask > 0.5, 0)
        return x 
