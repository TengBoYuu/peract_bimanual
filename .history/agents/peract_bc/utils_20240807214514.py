import numpy as np
import torch
import torch.nn.functional as F
import itertools
import gym
import h5py
import pandas as pd
import pickle
import cv2
import random
import string

import torch.distributions as pyd
from einops import rearrange, repeat
import math

def pad(x, max_len, axis=1, const=0, mode='pre'):
    """Pads input sequence with given const along a specified dim

    Inputs:
        x: Sequence to be padded
        max_len: Max padding length
        axis: Axis to pad (Default: 1)
        const: Constant to pad with (Default: 0)
        mode: ['pre', 'post'] Specifies whether to add padding pre or post to the sequence
    """

    if isinstance(x, tuple):
        x = np.array(x)

    pad_size = max_len - x.shape[axis]
    if pad_size <= 0:
        return x

    npad = [(0, 0)] * x.ndim
    if mode == 'pre':
        npad[axis] = (pad_size, 0)
    elif mode == 'post':
        npad[axis] = (0, pad_size)
    else:
        raise NotImplementedError

    if isinstance(x, np.ndarray):
        x_padded = np.pad(x, pad_width=npad, mode='constant', constant_values=const)
    elif isinstance(x, torch.Tensor):
        # pytorch starts padding from final dim so need to reverse chaining order
        npad = tuple(itertools.chain(*reversed(npad)))
        x_padded = F.pad(x, npad, mode='constant', value=const)
    else:
        raise NotImplementedError
    return x_padded

