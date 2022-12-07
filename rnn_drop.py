
import torch
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNNDrop(nn.Module):
    """
    RNNDrop implemnetation. 
    """

    def __init__(self, drop_prob):
        """
        Initializae dropout layer with a droput rate.
        @param drop_prob : Float value which marks the probability of dropping
                           each neuron.
        """
        super(RNNDrop, self).__init__()
        self.prob = drop_prob
    
    def reset_mask(self, mask_dim):
        """
        Given dimension creates a dropout mask for the input.
        Expected to be called at the beginning of each forward pass
        in BLSTM. So that each batch uses unique masks while
        mask of a single sequence stays same.
        @param mask_dim : Tuple of int specifying dimension of the mask.
        """
        prob = torch.ones(mask_dim).to(device) * (1-self.prob)
        q = torch.tensor(1/(1-self.prob)).to(device)
        self.mask = torch.bernoulli(prob).to(device) * q
        
    def forward(self, input):
        """
        Forward pass of RNN drop
        @param input : Input tensor
        @return Tensor after applying the dropout mask on the input values.
        """
        return self.mask * input
