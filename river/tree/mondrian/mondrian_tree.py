from abc import ABC
from river.tree.mondrian.mondrian_tree_nodes import MondrianTreeBranch


class MondrianTree(ABC):
    """
    Base class for Mondrian Trees.

    This is an **abstract class**, so it cannot be used directly. It defines base operations
    and properties that all the Mondrian Trees must inherit or implement according to
    their own design.

    Parameters
    ----------
    n_features
        Number of features
    step
        Step parameter of the tree
    loss
        Loss to minimize for each node of the tree
        Pick between: "log", ...
    use_aggregation
        Should it use aggregation
    split_pure
        Should the tree split pure leafs when training
    iteration
        Number of iterations to run when training
    """

    def __init__(
            self,
            n_features: int,
            step: float = 0.1,
            loss: str = "log",
            use_aggregation: bool = True,
            split_pure: bool = False,
            iteration: int = 0,
    ):
        # Properties common to all the Mondrian Trees
        self.n_features = n_features
        self.step = step
        self.loss = loss
        self.use_aggregation = use_aggregation
        self.split_pure = split_pure
        self.iteration = iteration
        self.intensities = [0] * n_features
        self.tree = None
