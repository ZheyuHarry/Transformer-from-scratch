"""
@author: Luo Yu
@time: 2024/06/01
"""

from torch import nn
from positional_encoding import PositionalEncoding
from token_embedding import TokenEmbedding

class TransformerEmbedding(nn.Module):
    """
    Token Embedding + Positional Encoding = Transformer Embedding
    positional encoding can give positional information to network
    dropout is used here for the reason that word embedding is trainable
    """

    def __init__(self , vocab_size , d_model , max_len , drop_prob , device):
        """
        Initialize a Transformer Embedding with all other parameters
        """
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size , d_model)
        self.position_encoding = PositionalEncoding(max_len,d_model , device)
        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self , x):
        """
        Calculate the token embeds and positional embeds individually.
        And then add them together to apply a dropout transformation
        """
        token_embeds = self.token_embedding(x)
        positional_embeds = self.position_encoding(x)
        out = self.dropout(token_embeds + positional_embeds)
        return out