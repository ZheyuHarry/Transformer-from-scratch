"""
@author: Luo Yu
@time: 2024/06/01
"""

from torch import nn

from layers.layer_norm import LayerNorm
from layers.multi_head_product_attention import MultiHeadAttention
from layers.position_wise_feed_forward_network import PositionWiseFeedForward

class EncoderLayer(nn.Module):
    """
    This is the construction of Encoder layer 
    """
    def __init__(self , d_model , n_heads , ffn_hidden , drop_prob):
        """
        @param d_model: dimensions of the model
        @param n_heads: number of heads in Multi-Head Attention
        @param ffn_hidden: number of hidden features in ffn hidden layer
        @param drop_prob: dropout probability
        """
        super(EncoderLayer, self).__init__()
        # Define two layers with each parameters
        self.attention = MultiHeadAttention(d_model , n_heads)
        self.FFN = PositionWiseFeedForward(d_model , ffn_hidden , drop_prob)

        # other normalization and dropout settings are the same
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(drop_prob)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self , x , src_mask):
        # 1. use residual connection here and compute attention
        r_x = x
        x = self.attention(q=x , k=x , v=x , mask=src_mask)

        # 2. Add and Norm
        x = self.dropout1(x)
        x = self.norm1(x + r_x)

        # 3. Calculate the Feed Forward network
        r_x = x
        x = self.FFN(x)

        # 4. Add and Norm
        x = self.dropout2(x)
        x = self.norm2(x + r_x)

        return x
