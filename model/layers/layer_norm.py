"""
@author: Luo Yu
@time: 2024/05/31
"""
import torch
from torch import nn

class LayerNorm(nn.Module):
    """
    Apply Normalization to the vectors
    The formula is x = gamma((x-u)/sqrt(var+epsilon)) + beta
    """
    def __init__(self , d_model , epsilon=1e-12):
        super(LayerNorm, self).__init__()
        # Pre-defined parameters gamma and beta
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = epsilon
    
    def forward(self , x):
        """
        x is the input tensor with shape of (batch_size , length , d_model) , and layernorm should be applied to the last dimension
        """
        # Here -1 means the last dimension , so the normalization is applied to each sample rather than feature
        print(f'X.size before layer norm is{x.size()}')
        mean = x.mean(-1 , keepdim=True)
        print(f'mean.size of input x is {mean.size()}')

        var = x.var(-1 , unbiased=False , keepdim=True)

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out