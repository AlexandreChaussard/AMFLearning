import numpy as np
from numpy import float32
from numpy import uint32
from numpy import uint8


def sample_discrete(distribution):
    """Samples according to the given discrete distribution.

    Parameters
    ----------
    distribution : `np.array', shape=(size,), dtype='float32'
        The discrete distribution we want to sample from. This must contain
        non-negative entries that sum to one.

    Returns
    -------
    output : `uint32`
        Output sampled in {0, 1, 2, distribution.size} according to the given
        distribution

    Notes
    -----
    It is useless to np.cumsum and np.searchsorted here, since we want a single
    sample for this distribution and since it changes at each call. So nothing
    is better here than simple O(n).

    Warning
    -------
    No test is performed here for efficiency: distribution must contain non-
    negative values that sum to one.
    """
    # Notes
    U = np.random.uniform(0.0, 1.0)
    cumsum = 0.0
    size = distribution.size
    for j in range(size):
        cumsum += distribution[j]
        if U <= cumsum:
            return j
    return size - 1


def resize_array(arr, keep, size, fill=0):
    """Resize the given array along the first axis only, preserving the same
    dtype and second axis size (if it's two-dimensional)

    Parameters
    ----------
    arr : `np.array`
        Input array

    keep : `int`
        Keep the first `keep` elements (according to the first axis)

    size : `int`
        Target size of the first axis of new array (

    fill : {`None`, 0, 1}, default=0
        Controls the values in the resized array before putting back the first elements
        * If None, the array is not filled
        * If 1 the array is filled with ones
        * If 0 the array is filled with zeros

    Returns
    -------
    output : `np.array`
        New array of shape (size,) or (size, arr.shape[1]) with `keep` first
        elements preserved (along first axis)
    """
    if arr.ndim == 1:
        if fill is None:
            new = np.empty((size,), dtype=arr.dtype)
        elif fill == 1:
            new = np.ones((size,), dtype=arr.dtype)
        else:
            new = np.zeros((size,), dtype=arr.dtype)
        new[:keep] = arr[:keep]
        return new
    elif arr.ndim == 2:
        _, n_cols = arr.shape
        if fill is None:
            new = np.empty((size, n_cols), dtype=arr.dtype)
        elif fill == 1:
            new = np.ones((size, n_cols), dtype=arr.dtype)
        else:
            new = np.zeros((size, n_cols), dtype=arr.dtype)
        new[:keep] = arr[:keep]
        return new
    else:
        raise ValueError("resize_array can resize only 1D and 2D arrays")


def log_sum_2_exp(a, b):
    """Computation of log( (e^a + e^b) / 2) in an overflow-proof way

    Parameters
    ----------
    a : `float32`
        First number

    b : `float32`
        Second number

    Returns
    -------
    output : `float32`
        Value of log( (e^a + e^b) / 2) for the given a and b
    """
    # TODO: if |a - b| > 50 skip
    # TODO: try several log and exp implementations
    if a > b:
        return a + np.log((1 + np.exp(b - a)) / 2)
    else:
        return b + np.log((1 + np.exp(a - b)) / 2)
