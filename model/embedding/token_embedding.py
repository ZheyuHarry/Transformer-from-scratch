"""
@author: Luo Yu
@time: 2024/06/01
"""

from torch import nn

class TokenEmbedding(nn.Embedding):
    """
    Using the module nn.Embedding to embed the tokens into vectors
    Utilizing a weighted matrix of shape (vocab_size , num_dimentions) , which can be updated with the model
    PS. I'm still a little confused about the padding_idx parameter
    """

    def __init__(self , vocab_size , d_model):
        """
        @param vocab_size: size of the vocabulary
        @param d_model: the number of dimensions of the model
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model,padding_idx=1)

