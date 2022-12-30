import typing

from abc import ABC, abstractmethod

from river.utils.mondrian_utils import log_sum_2_exp

from math import log

from river.tree.base import Leaf
from river.tree.base import Branch


class MondrianTreeLeaf(Leaf, ABC):
    """
    Abstract class for all types of nodes in a Mondrian Tree

    Parameters
    ----------
        parent: MondrianTreeLeaf
            Parent Node
        n_features: int
            Number of features of the input data
        time: float
            Split time of the node for Mondrian process

    Attributes
    ----------
        parent: MondrianTreeLeaf
            Parent Node
        time: float
            Split time of the node
        is_leaf: bool
            Says whether it's a leaf or parent node of some given children
        depth: int
            Depth of the node in the tree
        feature: int
            Feature index attached to the node
        weight: float
            Weight of the node
        log_weight_tree: float
            Log weight of the node in the tree
        threshold: float
            Threshold of the node to accept a sample
        n_samples: int
            Number of samples in the node
        n_features: int
            Number of features of the problem
        memory_range_min: list[float]
            Memory of the range of each feature seen by the node to compute its extension in the Mondrian process
        memory_range_max: list[float]
            Memory of the range of each feature seen by the node to compute its extension in the Mondrian process
    """

    def __init__(self, parent, n_features, time: float):

        super().__init__()

        # Generic Node attributes
        self.parent = parent
        self.time = time
        self.is_leaf = True
        self.depth = 0
        self._left = None
        self._right = None
        self.feature = 0
        self.weight = 0.0
        self.log_weight_tree = 0
        self.threshold = 0.0
        self.n_samples = 0
        self.n_features = n_features
        self.memory_range_min = [0] * n_features
        self.memory_range_max = [0] * n_features

    def copy(self, node):
        """
        Copies the node into the current one.
        Parameters
        ----------
        node: MondrianTreeLeaf
            Origin node to copy data from

        Returns
        -------

        """
        self.is_leaf = node.is_leaf
        self.depth = node.depth
        self.parent = node.parent
        self._left = node.get_left()
        self._right = node.get_right()
        self.feature = node.feature
        self.weight = node.weight
        self.log_weight_tree = node.log_weight_tree
        self.threshold = node.threshold
        self.time = node.time

    def set_left(self, node):
        """
        Set the left child node
        Parameters
        ----------
        node: MondrianTreeLeaf

        Returns
        -------

        """
        self._left = node

    def set_right(self, node):
        """
        Set the right child node
        Parameters
        ----------
        node

        Returns
        -------

        """
        self._right = node

    @abstractmethod
    def _get_child_node(self, node):
        """
        Get the child node (either left or right) and initialize it with default value if None exist
        This is meant to be used as a private function.
        Parameters
        ----------
        node: MondrianTreeLeaf

        Returns
        -------
            MondrianTreeLeaf
        """

    def get_left(self):
        """Get the left child"""
        return self._get_child_node(self._left)

    def get_right(self):
        """Get the right child"""
        return self._get_child_node(self._right)

    def update_depth(self, depth: int):
        """
        Updates the depth of the current node with the given depth
        Parameters
        ----------
        depth: int

        Returns
        -------

        """
        depth += 1
        self.depth = depth

        # if it's a leaf, no need to update the children too
        if self.is_leaf:
            return

        # Updating the depth of the children as well
        self._left = self.get_left()
        self._right = self.get_right()

        self._left.update_depth(depth)
        self._right.update_depth(depth)

    def update_weight_tree(self):
        """
        Updates the weight of the node in the tree
        Returns
        -------

        """
        if self.is_leaf:
            self.log_weight_tree = self.weight
        else:
            left = self.get_left()
            right = self.get_right()
            weight = self.weight

            self.log_weight_tree = log_sum_2_exp(
                weight, left.log_weight_tree + right.log_weight_tree
            )

    def get_child(self, x: dict):
        """
        Get child node classifying x properly
        Parameters
        ----------
        x: dict
            sample to find the path for

        Returns
        -------
            MondrianTreeLeaf

        """
        if x[self.feature] <= self.threshold:
            return self.get_left()
        else:
            return self.get_right()

    @property
    def __repr__(self):
        return f"Node : {self.parent}, {self.time}"


class MondrianTreeLeafClassifier(MondrianTreeLeaf):
    """
    Defines a node in a Mondrian Tree Classifier.

    Parameters
    ----------
        parent: MondrianTreeLeafClassifier
            Parent node
        n_features: int
            Number of features of the problem
        time: float
            Split time of the node
        n_classes: int
            Number of classes of the problem

    Attributes
    ----------
        n_classes: int
            Number of classes of the problem
        counts: list[int]
            List of amount of samples in the node belonging to each class

    """
    def __init__(self, parent, n_features: int, time: float, n_classes: int):
        super().__init__(parent, n_features, time)
        self.n_classes = n_classes
        self.counts = [0] * n_classes

    def _get_child_node(self, node):
        """
        Get the child node and initialize if none exists.
        Parameters
        ----------
        node: MondrianTreeLeafClassifier

        Returns
        -------

        """
        # If the child is None, it means we have to initialize it with default values, at the right depth
        # This is mostly to have material to work with during computations, rather than handling the None situation
        # separately each time we encounter it
        if node is None:
            node = MondrianTreeLeafClassifier(self, self.n_features, 0, self.n_classes)
            node.depth = self.depth + 1
        return node

    def score(self, sample_class: int, dirichlet: float):
        """
        Computes the score of the node

        Parameters
        ----------
        sample_class : int
            Class for which we want the score

        dirichlet : float
            Dirichlet parameter of the tree

        Returns
        -------
        output : float
            The log-loss of the node

        Notes
        -----
        This uses Jeffreys prior with dirichlet parameter for smoothing
        """

        count = self.counts[sample_class]
        n_classes = self.n_classes
        # We use the Jeffreys prior with dirichlet parameter
        return (count + dirichlet) / (self.n_samples + dirichlet * n_classes)

    def predict(self, dirichlet: float):
        """
        Predict the scores of all classes and updates the given `scores` dictionary with the new values

        Parameters
        ----------
        dirichlet: float
            Dirichlet parameter of the tree

        Returns
        -------
            scores: dict
        """
        scores = {}
        for c in range(self.n_classes):
            scores[c] = self.score(c, dirichlet)
        return scores

    def loss(self, sample_class, dirichlet):
        """
        Computes the loss of the node

        Parameters
        ----------
        sample_class: int
            A given class of the problem
        dirichlet: float
            Dirichlet parameter of the problem

        Returns
        -------
            loss: float
        """
        sc = self.score(sample_class, dirichlet)
        return -log(sc)

    def update_weight(self, sample_class, dirichlet, use_aggregation, step):
        """
        Updates the weight of the node given a class and the method used

        Parameters
        ----------
        sample_class: int
            Class of a given sample
        dirichlet: float
            Dirichlet parameter of the tree
        use_aggregation: bool
            Whether to use aggregation of not during computation (given by the tree)
        step: float
            Step parameter of the tree

        Returns
        -------
        loss: float
        """
        loss_t = self.loss(sample_class, dirichlet)
        if use_aggregation:
            self.weight -= step * loss_t
        return loss_t

    def update_count(self, sample_class):
        """
        Updates the amount of samples that belong to that class into the node (not to use twice if you add one sample)

        Parameters
        ----------
        sample_class: int
            Class of a given sample

        Returns
        -------

        """
        self.counts[sample_class] += 1

    def is_dirac(self, sample_class):
        """

        Says whether the node follows a dirac distribution regarding the given class.
        i.e. if the node is pure regarding the given class.

        Parameters
        ----------
        sample_class: int
            Class of a given sample

        Returns
        -------

        """
        return self.n_samples == self.counts[sample_class]

    def update_downwards(self, x_t, sample_class, dirichlet, use_aggregation, step, do_update_weight):
        """
        Updates the node when running a downward procedure updating the tree

        Parameters
        ----------
        x_t: list[float]
            Sample to proceed (as a list)
        sample_class:
            Class of the sample x_t
        dirichlet: float
            Dirichlet parameter of the tree
        use_aggregation: bool
            Should it use the aggregation or not
        step: float
            Step of the tree
        do_update_weight: bool
            Should we update the weights of the node as well

        Returns
        -------

        """
        # Updating the range of the feature values known by the node
        # If it is the first sample, we copy the features vector into the min and max range
        if self.n_samples == 0:
            for j in range(self.n_features):
                x_tj = x_t[j]
                self.memory_range_min[j] = x_tj
                self.memory_range_max[j] = x_tj
        # Otherwise, we update the range
        else:
            for j in range(self.n_features):
                x_tj = x_t[j]
                if x_tj < self.memory_range_min[j]:
                    self.memory_range_min[j] = x_tj
                if x_tj > self.memory_range_max[j]:
                    self.memory_range_max[j] = x_tj

        # One more sample in the node
        self.n_samples += 1

        if do_update_weight:
            self.update_weight(sample_class, dirichlet, use_aggregation, step)

        self.update_count(sample_class)

    def range(self, j: int) -> tuple[float, float]:
        """
        Outputs the known range of the node regarding the j-th feature
        Parameters
        ----------
        j: int
            Feature index for which you want to know the range

        Returns
        -------
        tuple[float, float]
        """
        return (
            self.memory_range_min[j],
            self.memory_range_max[j],
        )

    def range_extension(self, x_t, extensions):
        """
        Computes the range extension of the node for the given sample
        Parameters
        ----------
        x_t: list[float]
            Sample to deal with
        extensions: list[float]
            List of range extension per feature to update

        Returns
        -------
        extensions_sum: float
            Sum of all the extensions range
        """
        extensions_sum = 0
        for j in range(self.n_features):
            x_tj = x_t[j]
            feature_min_j, feature_max_j = self.range(j)
            if x_tj < feature_min_j:
                diff = feature_min_j - x_tj
            elif x_tj > feature_max_j:
                diff = x_tj - feature_max_j
            else:
                diff = 0
            extensions[j] = diff
            extensions_sum += diff
        return extensions_sum


class MondrianTreeBranch(Branch, ABC):
    """
    A generic branch implementation for a Mondrian Tree.
    parent and children are MondrianTreeLeaf objects

    Parameters
    ----------
        parent: MondrianTreeLeaf
            Origin node of the branch

    Attributes
    ----------
        parent: MondrianTreeLeaf
            Origin node of the branch
    """

    def __init__(self, parent: MondrianTreeLeaf):
        super().__init__((parent.get_left(), parent.get_right()))
        self.parent = parent

    def next(self, x) -> typing.Union["Branch", "Leaf"]:
        child = self.parent.get_child(x)
        if child.is_leaf:
            return child
        else:
            return MondrianTreeBranch(child)

    def most_common_path(self) -> typing.Tuple[int, typing.Union["Leaf", "Branch"]]:
        raise NotImplementedError

    @property
    def repr_split(self):
        raise NotImplementedError


class MondrianTreeBranchClassifier(MondrianTreeBranch):

    """
    A generic Mondrian Tree Branch for Classifiers.
    The specificity resides in the nature of the nodes which are all MondrianTreeLeafClassifier instances.

    Parameters
    ----------
        parent: MondrianTreeLeafClassifier
            Origin node of the tree
    """

    def __init__(self, parent: MondrianTreeLeafClassifier):
        super().__init__(parent)
        self.parent = parent

    def most_common_path(self) -> typing.Tuple[int, typing.Union["Leaf", "Branch"]]:
        raise NotImplementedError

    @property
    def repr_split(self):
        raise NotImplementedError
