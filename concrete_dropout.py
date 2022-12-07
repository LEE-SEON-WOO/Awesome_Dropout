import torch
from torch import nn

class ConcreteDropout(nn.Module):
    def __init__(self, p_logit=-2.0, temp=0.01, eps=1e-8):
        super(ConcreteDropout, self).__init__()
        self.p_logit = nn.Parameter(torch.Tensor([p_logit]))
        self.temp = temp
        self.eps = eps

    @property
    def p(self):
        return torch.sigmoid(self.p_logit)

    def forward(self, x):
        if self.train():
            unif_noise = torch.rand_like(x)
            drop_prob = torch.log(self.p + self.eps) -\
            torch.log(1-self.p + self.eps)+\
            torch.log(unif_noise + self.eps)-\
            torch.log(1-unif_noise + self.eps)
            drop_prob = torch.sigmoid(drop_prob/ self.temp)
            random_tensor = 1. - drop_prob
            retain_prob = 1. - self.p
            x *= random_tensor
            x /= retain_prob
        return x
