"""
@author: Luo Yu
@time: 2024/06/01
"""

from torch import nn

from layers.layer_norm import LayerNorm
from layers.multi_head_product_attention import MultiHeadAttention
from layers.position_wise_feed_forward_network import PositionWiseFeedForward

class DecoderLayer(nn.Module):
    """
    This is the construction of Decoder layer
    """
    def __init__(self , d_model , n_heads , ffn_hidden , drop_prob):
        super(DecoderLayer, self).__init__()

        self.attention1 = MultiHeadAttention(d_model,n_heads)
        self.attention2 = MultiHeadAttention(d_model,n_heads)
        self.ffn = PositionWiseFeedForward(d_model,ffn_hidden,drop_prob)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(drop_prob)
        self.dropout2 = nn.Dropout(drop_prob)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self , dec , enc , tgt_mask , src_mask):
        """
        @param dec: the decoder input embeddings
        @param enc: the encoder output vectors
        @param tgt_mask: the mask of the self-attention block
        @param src_mask: the mask of the encoder-decoder attention block
        @return: the output of the decoder layer
        """
        # 1. Calculate self-attention
        r_x = dec
        x = self.attention1(q = dec , k = dec , v = dec , mask=tgt_mask)
        # 2. Add and norm
        x = self.dropout1(x)
        x = self.norm1(x + r_x)

        # 3.Calculate encoder-decoder attention
        if enc is not None:
            r_x = x
            x = self.attention2(q = x , k = enc , v = enc , mask=src_mask)

            # 4. Add and norm
            x = self.dropout2(x)
            x = self.norm2(x + r_x)
        
        # 5. Calculate position-wise feed forward network
        r_x = x
        x = self.ffn(x)

        # 6. Add and norm
        x = self.dropout3(x)
        x = self.norm3(x + r_x)

        return x