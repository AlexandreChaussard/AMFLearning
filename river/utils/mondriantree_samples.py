# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
import numpy as np
from numpy import float32
from numpy import uint32
from numpy import uint8

from river.utils.mondrian_utils import resize_array

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

    def samples_collection_to_dict(self):
        d = {}
        for key, dtype in spec_samples_collection:
            d[key] = getattr(self, key)
        d["n_features"] = self.features.shape[1]
        return d

    def add_samples(self, X, y):
        """Adds the features `X` and labels `y` to the collection of samples `samples`

        Parameters
        ----------
        X : :obj:`np.ndarray`, shape=(n_samples, n_features)
            Input features matrix to be appended

        y : :obj:`np.ndarray`
            Input labels vector to be appended

        """
        n_new_samples, n_features = X.shape
        n_current_samples = self.n_samples
        n_samples_required = n_current_samples + n_new_samples
        capacity_missing = n_samples_required - self.n_samples_capacity
        if capacity_missing >= 0:
            # We don't have enough room. Increase the memory reserved.
            if capacity_missing > self.n_samples_increment:
                # If what's required is larger than the increment, we use what's missing
                # plus de minimum increment
                increment = capacity_missing + self.n_samples_increment
            else:
                increment = self.n_samples_increment

            n_samples_reserved = self.n_samples_capacity + increment
            self.features = resize_array(
                self.features, n_current_samples, n_samples_reserved
            )
            self.labels = resize_array(
                self.labels, n_current_samples, n_samples_reserved
            )
            self.n_samples_capacity += increment

        self.features[n_current_samples:n_samples_required] = X
        self.labels[n_current_samples:n_samples_required] = y
        self.n_samples += n_new_samples


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
