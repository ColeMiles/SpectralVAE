import numpy as np
import torch

from specvae.data_util import data_util


def test_make_splits():
    length = 100
    split_fracs = (0.7, 0.2, 0.1)
    for _ in range(5):
        splits = data_util._make_splits(length, split_fracs=split_fracs)

        # Check all splits are of the same length
        len_train = int(np.ceil(length * split_fracs[0]))
        len_val = int(np.ceil(length * split_fracs[1]))
        len_test = length - len_train - len_val
        assert len(splits[0]) == len_train
        assert len(splits[1]) == len_val
        assert len(splits[2]) == len_test

        # Generate the splits again, check that we get the same splits
        splits_again = data_util._make_splits(length, split_fracs=split_fracs)
        assert torch.equal(splits[0], splits_again[0])
        assert torch.equal(splits[1], splits_again[1])
        assert torch.equal(splits[2], splits_again[2])
