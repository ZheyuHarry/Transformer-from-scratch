"""
@author: Luo Yu
@time: 2024/05/31
"""

import torch 
from torch import nn

class PositionalEncoding(nn.Module):
    """
    We use positional encoding to give the input vectors information on posititon
    """
    def __init__(self , max_len , d_model ,device):
        """
        @param max_len: maximum length of input sequence
        @param d_model: dimension of input vectors
        @param device: the hardware device
        """
        super(PositionalEncoding, self).__init__()

        # 1. Initialize positional encoding matrix with the same shape of input vectors
        self.encoding = torch.zeros(max_len, d_model , device=device)
        self.encoding.requires_grad = False # We don't need to calculate the gradients

        # 2. Initialize the positions , shape = (max_len , 1)
        pos = torch.arange(0 , max_len , device=device) # This is 1D tensor of max_len like [0 , 1 , 2, 3]
        pos = pos.float().unsqueeze(dim=1) # This is now a 2D tensor where each one of its rows is a position like [[0] , [1] , [2] , [3]]

        # 3. Initialize the even number of positions , shape = (d_model // 2)
        _2i = torch.arange(0 , d_model , step = 2 ,device=device).float()

        # 4. Calculate the encoding matrix with sin & cos.
        # Here we also use broadcasting to calculate the encoding matrix.
        self.encoding[: , 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[: , 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self , x):
        """
        x is not directly added here
        """
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, length = x.size()
        # [batch_size = 128, length = 30]

        # return the first length's positional encoding rows
        return self.encoding[:length, :]
        # [length = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]
