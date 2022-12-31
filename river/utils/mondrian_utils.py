from random import uniform
from random import choices
from math import log
from math import exp
from math import fsum


def sample_discrete(distribution):
    """
    Samples an index according to the given discrete distribution
    Parameters
    ----------
    distribution: list[float]
        Weights of the distribution at each given index

    Returns
    -------
    index: int
    """
    size = len(distribution)
    indexes = list(range(size))
    return choices(indexes, distribution, k=1)[0]


def old_sample_discrete(distribution):
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
    U = uniform(0.0, 1.0)
    cumsum = 0.0
    size = len(distribution)
    for j in range(size):
        cumsum += distribution[j]
        if U <= cumsum:
            return j
    return size - 1


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
        return a + log((1 + exp(b - a)) / 2)
    else:
        return b + log((1 + exp(a - b)) / 2)
