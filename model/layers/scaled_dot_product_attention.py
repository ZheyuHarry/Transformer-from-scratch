"""
@author: Luo Yu
@time: 2024/05/31
"""

import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    """
    Implementation of the ScaledDotProductAttention
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax - nn.Softmax(dim=-1)

    def forward(self , q , k , v , mask = None , e=1e-12):
        """
        @param q,k,v: q,k,v are the corresponding tensor calculated by W_Q,W_K,W_V and 
        q,k,v are four dimensional tensors -> [batch_size , head , length , d_k]

        @param mask: mask is a matrix of the same shape as attention score indicating whether to mask the Attention
        length is the number of tokens in the sequence & d_k is the number of dimensions of tensor k
        """
        batch_size , head , length , d_k = k.size()

        # 1.Transpose the matrix k
        k_t = k.transpose(2,3)

        # 2.Calculate the query with key
        score = q @ k_t / math.sqrt(d_k) # scaled

        # 3.Optional masking
        # Here "mask == 0" will get a booling matrix with true and false , because mask is a matrix of 0 & 1 indicating the masked position
        if mask is not None:
            score = score.masked_fill(mask == 0 , -10000)

        # 4.calculate the new values of the input sequence
        v = score @ v

        return v , score
