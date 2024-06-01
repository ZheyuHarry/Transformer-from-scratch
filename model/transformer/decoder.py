"""
@author: Luo Yu
@time: 2024/06/01
"""

from torch import nn

from blocks.decoder_layer import DecoderLayer
from embedding.transformer_embedding import TransformerEmbedding

class Decoder(nn.Module):
    """
    This is the construction of Decoder
    """
    def __init__(self , d_model , n_heads , ffn_hidden , max_len , dec_voc_size , n_layers , drop_prob , device):
        self.embeds = TransformerEmbedding(vocab_size=dec_voc_size , d_model=d_model,max_len=max_len,drop_prob=drop_prob,device=device)
        self.layers = nn.ModuleList([DecoderLayer(d_model , n_heads , ffn_hidden , drop_prob) for i in range(n_layers)])
        self.linear = nn.Linear(d_model , dec_voc_size)
    
    def forward(self , tgt , enc_src , tgt_mask , src_mask):
        # 1.Embeds the target sequence
        x = self.embeds(tgt)

        for layer in self.layers:
            x = layer(x , enc_src , tgt_mask , src_mask)

        x = self.linear(x)
        return x

