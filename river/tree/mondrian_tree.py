import collections
import functools
import io
import math
import typing
from abc import ABC, abstractmethod

import numpy as np

from river.tree.nodes.mondriantree_nodes import *
from river.tree.nodes.mondriantree_utils import SamplesCollection

spec_tree = [
    ("n_features", uint32),
    ("step", float32),
    ("loss", None),
    ("use_aggregation", bool),
    ("split_pure", bool),
    ("samples", SamplesCollection),
    ("intensities", np.array),
    ("iteration", uint32),
]


class MondrianTree(ABC):
    """Base class for Mondrian Trees.


    This is an **abstract class**, so it cannot be used directly. It defines base operations
    and properties that all the Mondrian Trees must inherit or implement according to
    their own design.

    Parameters
    ----------
        n_features,
        step,
        loss,
        use_aggregation,
        split_pure,
        samples,
        iteration,
        n_nodes,
        n_nodes_capacity
    """

    def __init__(
            self,
            n_features: int,
            step: float,
            loss,
            use_aggregation: bool,
            split_pure: bool,
            samples: SamplesCollection,
            iteration: int,
            n_nodes: int,
            n_nodes_capacity: int,
    ):
        # Properties common to all the Mondrian Trees
        self.n_features = n_features
        self.step = step
        self.loss = loss
        self.use_aggregation = use_aggregation
        self.split_pure = split_pure
        self.samples = samples
        self.iteration = iteration
        self.n_nodes = n_nodes
        self.n_nodes_capacity = n_nodes_capacity
        self.intensities = np.empty(n_features, dtype=np.float32)
