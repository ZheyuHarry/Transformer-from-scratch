"""
@author: Luo Yu
@time: 2024/06/01
"""

from torch import nn
from blocks.encoder_layer import EncoderLayer
from embedding.transformer_embedding import TransformerEmbedding

class Encoder(nn.Module):
    """
    This is the construction of encoder
    """
    def __init__(self , enc_voc_size , max_len , d_model , n_heads , ffn_hidden , n_layers , drop_prob , device):
        """
        All of these parameters are essential to construct the encoder layer
        """
        super().__init__()
        # Use Transformer embedding which compose of token embedding & positional embedding
        self.embeds = TransformerEmbedding(vocab_size=enc_voc_size, max_len=max_len, d_model=d_model, device=device , drop_prob=drop_prob)

        # We have n_layers of EncoderLayers
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, n_heads=n_heads, ffn_hidden=ffn_hidden, drop_prob=drop_prob) for i in range(n_layers)])

    def forward(self , x , src_mask):
        x = self.embeds(x)
        for layer in self.layers:
            x = layer(x , src_mask)
        return x