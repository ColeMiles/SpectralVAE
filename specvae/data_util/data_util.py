#!/usr/bin/env python3

import os
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from specvae.utils.logger import log
from specvae.utils.timing import time_func


def _find_target_files(directory: str) -> List[str]:
    """ Given a directory, returns absolute paths of all of the contained
         .csv files containing `targets` in the name.
    """
    return [os.path.abspath(os.path.join(directory, p))
            for p in os.listdir(directory) if "targets" in p]


def _load_data(data_files: Union[str, List[str]]) -> torch.Tensor:
    """ Given a list of source data files, reads all data into a single
         torch.Tensor.
    """
    data_files = list(data_files)

    data = []
    for filename in data_files:
        data.append(pd.read_csv(filename, index_col=0).to_numpy())

    # Check that all data has the same shape, except for the first axis
    for arr in data[1:]:
        assert arr.shape[1:] == data[0].shape[1:]

    return torch.from_numpy(np.concatenate(data, axis=0)).float()


def _make_splits(
    length: int, split_fracs=(0.8, 0.1, 0.1)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Given the number of data points, generates the indices for a train/val/test split
         of the given fractional proportions. Splits deterministic dependent on the seed.
    """
    # Generate splits by slicing the first fractions of the data
    idxs = np.arange(length)
    t_idx = int(np.ceil(split_fracs[0] * length))
    v_idx = t_idx + int(np.ceil(split_fracs[1] * length))
    train_idxs = torch.as_tensor(idxs[:t_idx])
    val_idxs = torch.as_tensor(idxs[t_idx:v_idx])
    test_idxs = torch.as_tensor(idxs[v_idx:])

    return train_idxs, val_idxs, test_idxs


@time_func(log)
def prepare_autoencoder_data(
    directory: str, split_fracs=(0.8, 0.1, 0.1),
    batch_size=32, conv=False, indices=None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """ Given a directory containing data files, produces three DataLoaders for
         a train/val/test split of the data. These loaders are intended for an
         autoencoder usage, so they only produce a single tensor on iteration:
         the spectral function.
        If explicit train/val/test indices are not provided, just deterministically
         splits the data based on split_fracs.
    """
    data = _load_data(_find_target_files(directory))
    # Add a "channel" dimension if this data is intended for a convolutional arch
    if conv:
        data = data.reshape(data.shape[0], 1, data.shape[1])

    if indices is not None:
        train_idxs = indices['train']
        val_idxs = indices['val']
        test_idxs = indices['test']
    else:
        train_idxs, val_idxs, test_idxs = _make_splits(len(data), split_fracs=split_fracs)

    train_dataset = TensorDataset(data[train_idxs])
    val_dataset = TensorDataset(data[val_idxs])
    test_dataset = TensorDataset(data[test_idxs])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             pin_memory=False)

    return train_loader, val_loader, test_loader
