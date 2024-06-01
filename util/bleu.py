"""
@author: Luo Yu
@time: 2024/06/01
"""

import math
from collections import Counter
import numpy as np

def bleu_stats(hypothesis , reference):
    """
    Compute statistics for bleu
    """
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))

    for n in range(1,5):
        s_ngrams = Counter(
            [tuple(hypothesis[i : i + n]) for i in range(len(hypothesis) + 1 - n)]
        )

        r_ngrams = Counter(
            [tuple(reference[i : i + n]) for i in range(len(reference) + 1 - n)]
        )
        # Add the numbers of n-grams that match the reference
        stats.append(max(sum((s_ngrams & r_ngrams).values() , 0)))
        # Add the numbers of n-grams
        stats.append(max(len(hypothesis) + 1 - n , 0))

    return stats    

def BLEU(stats):
    """
    Compute BLEU score given n-gram stats
    """
    # If the number of intersection is less than zero , the BLEU should be 0
    if len(list(filter(lambda x : x==0 , stats))) > 0:
        return 0
    
    # get the length of hypothesis and reference
    (c , r) = stats[:2]

    # get the average log accuracy of the hypothesis and reference
    log_bleu_prec = sum([math.log(float(x) / y) for x , y in zip(stats[2::2] , stats[3::2])]) / 4

    # get the brevity penalty with function min , if r >> c , then bp will be far less than 0
    bp = min(0 , 1 - float(r)/c)
    bleu_score = math.exp(bp + log_bleu_prec)
    return bleu_score

def get_bleu(hypotheses , references):
    """
    Get the bleu score 
    """
    # Here we set 10 elements in stats is because we need c and r , and 4 n-grams with 2 elements each , summing up to 10
    stats = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Calculate all the statistics data
    for h , r in zip(hypotheses , references):
        stats += np.array(bleu_stats(h,r))

    return BLEU(stats) * 100


def idx_to_word(x, vocab):
    """
    Given a index list , return a sentence with corresponding words
    """
    words = []
    for i in x:
        word = vocab.itos[i]
        if "<" not in word:
            words.append(word)
    words = " ".join(words)
    return words
