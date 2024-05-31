"""
@author: Luo Yu
@time: 2024/05/31
"""

from torch import nn

class PositionWiseFeedForward(nn.Module):
    """
    This layer is used to convert the vectors after the attention layer to a different Semantic space.
    """
    def __init__(self , d_model , hidden , drop_prob=0.01):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model , hidden)
        self.linear2 = nn.Linear(hidden , d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
    
    def forward(self , x):
        """
        We use two linear transformation and a ReLU activation in betweens
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x