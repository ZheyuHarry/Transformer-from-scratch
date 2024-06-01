"""
@author: Luo Yu
@time: 2024/06/01
"""
import torch
from torch import nn

from decoder import Decoder
from encoder import Encoder

class Transformer(nn.Module):
    """
    The Final model
    """
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_heads, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx

        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_heads=n_heads,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_heads=n_heads,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self , src , tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_src = self.encoder(src , src_mask)
        output = self.decoder(tgt , enc_src , tgt_mask , src_mask)
        return output

    def make_src_mask(self, src):
        """
        Calculate pad_mask to avoid unnecessary computation at pad_idx
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        """
        Calculate pad_mask to avoid unnecessary computation at pad_idx , and calculate sub_mask to let element at position i can only "see" the pos between [1,i]
        And add them together
        """
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(3)
        tgt_len = tgt.shape[1]
        tgt_sub_mask = (
            torch.tril(torch.ones(tgt_len, tgt_len))
            .type(torch.ByteTensor)
            .to(self.device)
        )
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask
