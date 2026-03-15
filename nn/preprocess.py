# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # split labels into positive and negative 
    positive = []
    negative = []
    for seq, label in zip(seqs,labels):
        if label == True:
            positive.append(seq)
        elif label == False:
            negative.append(seq)
    # balancing the two by finding the length of negative and sampling the positive to match
    sampled_pos_idx = np.random.choice(len(positive), size = len(negative), replace = True)
    sampled_pos = []
    for index in sampled_pos_idx:
        sampled_pos.append(positive[index])
    
    # concatenate two lists
    sampled_seqs = sampled_pos + negative
    sampled_labels = [True] * len(sampled_pos) + [False] * len(negative)

    return (sampled_seqs, sampled_labels)

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    encoding_map = {
    'A': [1, 0, 0, 0],
    'T': [0, 1, 0, 0],
    'C': [0, 0, 1, 0],
    'G': [0, 0, 0, 1]
    }
    # for each sequence in seq_arr loop through character and look up encoding and add encoding to one long array

    encodings = []
    for sequence in seq_arr:
        sequence_encode = []
        for character in sequence:
            sequence_encode.extend(encoding_map[character])  
        encodings.append(sequence_encode)  

    return np.array(encodings)
        