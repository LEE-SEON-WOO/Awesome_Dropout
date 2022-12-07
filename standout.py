""" StandOut
https://github.com/mabirck/adaptative-dropout-pytorch

L. J. Ba and B. Frey, Adaptive dropout for training deep neural networks
"""
import torch
from torch.autograd import Variable
from torch import nn

class Standout(nn.Module):
    """
    Standout Layer:
    We define the Standout Layer here (as per Algorithm 2 in the original paper).
    It inherits from nn.Module class of PyTorch. The Standout Layer can easily 
    be converted to a standard Dropout layer by setting paramaeter alpha=0 and beta=0.5 for a Dropout rate of 0.5.
    """
    def __init__(self, last_layer:nn.Module, alpha:float, beta:float):
        super(Standout, self).__init__()
        self.pi = last_layer.weight
        self.alpha = alpha
        self.beta = beta
        self.nonlinearity = nn.Sigmoid() # Sigmoid used in the original paper

    # Forward propagation via the layer
    def forward(self, previous, current, deterministic=False):
        # Function as in page 3 of paper: Variational Dropout
        self.p = self.nonlinearity(self.alpha * previous.matmul(self.pi.t()) + self.beta)
        self.mask = sample_mask(self.p)

        # Deterministic version as in the paper
        if(deterministic or torch.mean(self.p).data.cpu().numpy()==0):
            return self.p * current
        else:
            return self.mask * current

def sample_mask(p):
    """Given a matrix of probabilities, this will sample a mask in PyTorch.
    Sampling Operation:
    Now we perform the sampling operation for the dropout. We pass a Tensor of Retention Probabilities 
    i.e the probability with which a node will be retained, and the function returns a Tensor of the same size as Retenetion Probabilities 
    which we call the Mask. The Mask Tensor contains 0 and 1 values where 1 indicates a node is retained and 0 indicates a node is dropped.
    """
    
    #Random Sampling
    if torch.cuda.is_available():
        uniform = Variable(torch.Tensor(p.size()).uniform_(0, 1).cuda())
    else:
        uniform = Variable(torch.Tensor(p.size()).uniform_(0, 1))
    #Setting Mask
    mask = uniform < p
    #Setting proper Data Type
    if torch.cuda.is_available():
        mask = mask.type(torch.cuda.FloatTensor)
    else:
        mask = mask.type(torch.FloatTensor)
    return mask