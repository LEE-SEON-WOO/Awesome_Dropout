
import torch
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GradBasedDropout(nn.Module):
    """
    Gradient Based Dropout layer.
    """
    
    def __init__(self, input_dim, drop_prob):
        """
        Initializes Grad Based Dropout layer.
        @param input_dim : Dimenson of keep probability (same as dimenson of layer)
        @param drop_prob : Dropout rate. Can be None as well when scaling to 0-1 range.
        """
        super(GradBasedDropout, self).__init__()
        self.keep_prob = torch.ones(input_dim).to(device)
        self.drop_prob = drop_prob
    
    def update_keep_prob(self, grad, method):
        """
        Given gradient and method scale gradients to a value within
        0 - 1. If method is TANH absolute of grad is taken and tanh
        function is applied. If ABS_NORM then absolute of gradient
        is taken and scaled within a 0-1 range.
        If dropout rate is not None then value is scaled to
        (1-drop_prob) to 1. So probabily of keeping neuron is higher.
        """
        grad = torch.abs(grad).sum(dim=-1)

        if method == "TANH":
            self.keep_prob = torch.tanh(grad)
        elif method == "ABS_NORM":
            self.keep_prob = (grad - torch.min(grad))/(torch.max(grad) - torch.min(grad) + 1e-7)
        
        if self.drop_prob is not None:
            self.keep_prob = (self.keep_prob * self.drop_prob) + (1-self.drop_prob)

    def forward(self, x):
        """
        Forward pass of gradient based dropout.
        @param x : Input tensor
        
        @return Tensor after applying dropout mask on the input tensor.
        """
        # for all batches same probability
        keep_prob = torch.ones(x.shape).to(device) * torch.unsqueeze(self.keep_prob, dim=0) 
        keep_prob = torch.clip(keep_prob, min=0.00001, max=1)
        mask = torch.bernoulli(keep_prob) * 1/keep_prob
        return mask * x