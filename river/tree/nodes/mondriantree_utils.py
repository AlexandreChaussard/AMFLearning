# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
import numpy as np
from numpy import float32
from numpy import uint32
from numpy import uint8

spec_samples_collection = [
    ("features", np.array),
    ("labels", np.array),
    ("n_samples_increment", uint32),
    ("n_samples", uint32),
    ("n_samples_capacity", uint32),
]


class SamplesCollection(object):
    """A class which simply keeps in memory the samples used for training when
        using repeated call to ``partial_fit``. A minimum increment is used when
        extending the capacity of the collection, in order to avoid repeated copies
        when ``add_samples`` is used on small batches.

    Attributes
    ----------
    n_samples_increment : :obj:`int`
        The minimum amount of memory which is pre-allocated each time extra memory is
        required for new samples.

    n_samples : :obj:`int`
        Number of samples currently saved in the collection.

    n_samples_capacity : :obj:`int`
        Number of samples that can be currently saved in the object.

    Note
    ----
    This is not intended for end-users. No sanity checks are performed here, we assume
    that such tests are performed beforehand in objects using this class.
    """

    def __init__(self, n_samples_increment, n_features, n_samples, n_samples_capacity):
        """Instantiates a `SamplesCollection` instance.

        Parameters
        ----------
        n_samples_increment : :obj:`int`
            Sets the amount of memory which is pre-allocated each time extra memory is
            required for new samples.

        n_features : :obj:`int`
            Number of features used during training.
        """
        if n_samples == 0:
            self.n_samples_increment = n_samples_increment
            self.n_samples_capacity = n_samples_increment
            self.features = np.empty((n_samples_increment, n_features), dtype=float32)
            self.labels = np.empty(n_samples_increment, dtype=uint32)
            self.n_samples = 0
        else:
            self.n_samples_increment = n_samples_increment
            self.n_samples_capacity = n_samples_capacity
            self.n_samples = n_samples
            self.features = np.empty((n_samples_capacity, n_features), dtype=float32)
            self.labels = np.empty(n_samples_capacity, dtype=float32)


def samples_collection_to_dict(samples):
    d = {}
    for key, dtype in spec_samples_collection:
        d[key] = getattr(samples, key)
    d["n_features"] = samples.features.shape[1]
    return d


def dict_to_samples_collection(d):
    n_samples_increment = d["n_samples_increment"]
    n_samples_capacity = d["n_samples_capacity"]
    n_samples = d["n_samples"]
    n_features = d["n_features"]
    samples = SamplesCollection(
        n_samples_increment, n_features, n_samples, n_samples_capacity
    )
    samples.features[:] = d["features"]
    samples.labels[:] = d["labels"]
    return samples


def add_samples(samples, X, y):
    """Adds the features `X` and labels `y` to the collection of samples `samples`

    Parameters
    ----------
    samples : :obj:`SamplesCollection`
        The collection of samples where we want to append X and y

    X : :obj:`np.ndarray`, shape=(n_samples, n_features)
        Input features matrix to be appended

    y : :obj:`np.ndarray`
        Input labels vector to be appended

    """
    n_new_samples, n_features = X.shape
    n_current_samples = samples.n_samples
    n_samples_required = n_current_samples + n_new_samples
    capacity_missing = n_samples_required - samples.n_samples_capacity
    if capacity_missing >= 0:
        # We don't have enough room. Increase the memory reserved.
        if capacity_missing > samples.n_samples_increment:
            # If what's required is larger than the increment, we use what's missing
            # plus de minimum increment
            increment = capacity_missing + samples.n_samples_increment
        else:
            increment = samples.n_samples_increment

        n_samples_reserved = samples.n_samples_capacity + increment
        samples.features = resize_array(
            samples.features, n_current_samples, n_samples_reserved
        )
        samples.labels = resize_array(
            samples.labels, n_current_samples, n_samples_reserved
        )
        samples.n_samples_capacity += increment

    samples.features[n_current_samples:n_samples_required] = X
    samples.labels[n_current_samples:n_samples_required] = y
    samples.n_samples += n_new_samples


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
