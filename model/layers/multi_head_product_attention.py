"""
@author: Luo Yu
@time: 2024/05/31
"""

import torch.nn as nn
from scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Implementation of MultiHeadAttention
    """
    def __init__(self , d_model , n_heads):
        """
        @param d_model: d_model is the length of the word vector after word embedding
        @param n_heads: number of heads in attention mechanism
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.attention = ScaledDotProductAttention()
        self.W_q = nn.linear(d_model, d_model)
        self.W_k = nn.linear(d_model, d_model)
        self.W_v = nn.linear(d_model, d_model)
        self.proj = nn.Linead(d_model , d_model)
        self.n_heads = n_heads

    def split(self ,  tensor):
        """
        split tensor by number of heads

        @param tensor: tensor shape [batch_size , length , d_model]
        @return: tensor shape [batch_size , n_heads , length , d_tensor]
        """

        batch_size , length , d_tensor = tensor.size()
        return tensor.view(batch_size , length , self.n_heads , d_tensor // self.n_heads).transpose(1 , 2)

    def concat(Self , tensor):
        """
        inverse function of self,split

        @param tensor: tensor shape  [batch_size , n_heads , length , d_tensor]
        @return: tensor shape [batch_size , length , d_model]
        """

        batch_size , heads , length , d_tensor = tensor.size()
        return tensor.transpose(1,2).contiguous().view(batch_size , length , heads * d_tensor)

    def forward(self , q , k , v , mask = None):
        """
        Calculate the multi-heads attention

        @param q,k,v : Query , Key , Value word vectors
        @param mask: Masking 
        """

        # 1.get the q , k , v vector
        q , k , v = self.W_q(q) , self.W_k(k) , self.W_v(v)

        # 2.split the q , k , v vector
        q , k , v = self.split(q) , self.split(k) , self.split(v)

        # 3.calculate the attention score and corresponding new v
        out , score = self.attention_(q,k,v,mask)

        # 4.concat the output tensor to only one head
        out = self.concat(out)
        
        # 5.project the concat tensor to a new space
        out = self.proj(out)

        return out