from abc import ABC, abstractmethod

from river.tree.nodes.mondrian_tree_nodes_riverlike import *
from river.utils.mondriantree_samples import SamplesCollection


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
    ):
        # Properties common to all the Mondrian Trees
        self.n_features = n_features
        self.step = step
        self.loss = loss
        self.use_aggregation = use_aggregation
        self.split_pure = split_pure
        self.samples = samples
        self.iteration = iteration
        self.intensities = np.empty(n_features, dtype=np.float32)
        self.tree = None
