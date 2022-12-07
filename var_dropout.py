import torch
from torch import nn
"""

"""
#https://github.com/j-min/Dropouts/blob/master/Gaussian_Variational_Dropout.ipynb
class VariationalDropout(nn.Module):
    def __init__(self, log_alpha=-3.):
        super(VariationalDropout, self).__init__()
        self.max_log_alpha = 0.0
        self.log_alpha = nn.Parameter(torch.Tensor([log_alpha]))
        
    @property
    def alpha(self):
        return torch.exp(self.log_alpha)
        
    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            normal_noise = torch.randn_like(x)
            self.log_alpha.data = torch.clamp(self.log_alpha.data, max=self.max_log_alpha)
            random_tensor = 1. + normal_noise * torch.sqrt(self.alpha)
            x *= random_tensor
        return x

