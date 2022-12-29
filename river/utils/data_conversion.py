import numpy as np


def dict2numpy(data) -> np.ndarray:
    """Convert a dictionary containing data to a numpy array.

    There is not restriction to the type of keys in `data`, but values must
    be strictly numeric. To make sure random permutations of the features
    do not impact on the learning algorithms, keys are first converted to
    strings and then sorted prior to the conversion.

    Parameters
    ----------
    data
        A dictionary whose keys represent input attributes and the values
        represent their observed contents.

    Returns
    -------
    An array representation of the values in `data`.

    """
    data_ = {str(k): v for k, v in data.items()}
    return np.asarray(list(x for _, x in sorted(data_.items())))


def numpy2dict(data: np.ndarray) -> dict:
    """Convert a numpy array to a dictionary.

    Parameters
    ----------
    data
        An one-dimensional numpy.array.

    Returns
    -------
    A dictionary where keys are integers $k \\in \\left{0, 1, ..., |\\text{data}| - 1\\right}$,
    and the values are each one of the $k$ entries in `data`.

    Examples
    --------

    """
    return {k: v for k, v in enumerate(data)}
