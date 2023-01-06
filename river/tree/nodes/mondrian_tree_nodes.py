import typing

from abc import ABC, abstractmethod

from river.utils.mondrian_utils import *

from river.tree.base import Leaf
from river.tree.base import Branch


class MondrianTreeLeaf(Leaf, ABC):
    """In a Mondrian Tree, a node is defined by its index, its parent index (can be None) and its creation time.
       The index is given by the tree, while the parent index and the creation time are variables of the node
    """

    def __init__(self, parent, n_features, time: float):
        """

        Parameters
        ----------
        parent: MondrianTreeLeaf
            Parent Node
        n_features: int
            Number of features of the input data
        time: float
            Split time of the node for Mondrian process
        """

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
        Get the child node and initialize it with default value if None exist
        Parameters
        ----------
        node: MondrianTreeLeaf

        Returns
        -------

        """

    def get_left(self):
        """Get the left child"""
        return self._get_child_node(self._left)

    def get_right(self):
        """Get the right child"""
        return self._get_child_node(self._right)

    def update_depth(self, depth: int):
        """
        Update the depth of the current node with the given depth
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
            sample to classify

        Returns
        -------

        """
        if x[self.feature] <= self.threshold:
            return self.get_left()
        else:
            return self.get_right()

    @property
    def __repr__(self):
        return f"Node : {self.parent}, {self.time}"


class MondrianTreeLeafClassifier(MondrianTreeLeaf):

    def __init__(self, parent, n_features, time: float, n_classes: int):
        super().__init__(parent, n_features, time)
        self.n_classes = n_classes
        self.counts = np.zeros(n_classes)

    def _get_child_node(self, node):
        """
        Get the child node and initialize if none exists.
        Parameters
        ----------
        node: MondrianTreeLeafClassifier

        Returns
        -------

        """
        if node is None:
            node = MondrianTreeLeafClassifier(self, self.n_features, 0, self.n_classes)
            node.is_leaf = True
            node.depth = self.depth + 1
        return node

    def score(self, sample_class: int, dirichlet: float):
        """Computes the score of the node

        Parameters
        ----------
        sample_class : int
            Class for which we want the score

        dirichlet : float

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

    def predict(self, dirichlet: float, scores: dict):
        """
        Predict the scores of all classes
        Parameters
        ----------
        dirichlet: float
        scores: float
            Dictionary of scores to be updated with the scores values

        Returns
        -------

        """
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
        dirichlet

        Returns
        -------

        """
        sc = self.score(sample_class, dirichlet)
        return -np.log(sc)

    def update_weight(self, sample_class, dirichlet, use_aggregation, step):
        """
        Update the weight of the node given a class and the method used
        Parameters
        ----------
        sample_class
        dirichlet
        use_aggregation
        step

        Returns
        -------

        """
        loss_t = self.loss(sample_class, dirichlet)
        if use_aggregation:
            self.weight -= step * loss_t
        return loss_t

    def update_count(self, sample_class):
        self.counts[sample_class] += 1

    def is_dirac(self, y_t):
        c = y_t
        n_samples = self.n_samples
        count = self.counts[c]
        return n_samples == count

    def update_downwards(self, x_t, sample_class, dirichlet, use_aggregation, step, do_update_weight):
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

    def range(self, j):
        return (
            self.memory_range_min[j],
            self.memory_range_max[j],
        )

    def range_extension(self, x_t, extensions):
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


class MondrianTreeLeafRegressor(MondrianTreeLeaf):

    def __init__(self, parent, n_features, time: float):
        super().__init__(parent, n_features, time)
        self.mean = np.zeros()

    def _get_child_node(self, node):
        """
        Get the child node and initialize if none exists.
        Parameters
        ----------
        node: MondrianTreeLeafRegressor

        Returns
        -------

        """
        if node is None:
            node = MondrianTreeLeafRegressor(self, self.n_features, 0)
            node.is_leaf = True
            node.depth = self.depth + 1
        return node

    def predict(self):
        """Returns the prediction of the node.
        Parameters
        ----------
        Returns
        -------
        Notes
        -----
        This uses Jeffreys prior with dirichlet parameter for smoothing
        """
        return self.mean

    def loss(self, sample_class):
        r = self.predict(self) - sample_class
        return r * r / 2

    def update_weight(self, sample_class, use_aggregation, step):
        loss_t = self.loss(self, sample_class)
        if use_aggregation:
            self.weight -= step * loss_t
        return loss_t

    def update_downwards(self, x_t, y_t, sample_class, use_aggregation, step, do_update_weight):
        if self.n_samples == 0:
            for j in range(self.n_features):
                x_tj = x_t[j]
                self.memory_range_min[j] = x_tj
                self.memory_range_max[j] = x_tj
        else:
            for j in range(self.n_features):
                x_tj = x_t[j]
                if x_tj < self.memory_range_min[j]:
                    self.memory_range_min[j] = x_tj
                if x_tj > self.memory_range_max[j]:
                    self.memory_range_max[j] = x_tj

        self.n_samples += 1

        if do_update_weight:
            self.update_weight(self, sample_class, use_aggregation, step)

        # Update the mean of the labels in the node online
        self.mean = (self.n_samples * self.mean + y_t) / (self.n_samples + 1)

    def range(self, j):
        return (
            self.memory_range_min[j],
            self.memory_range_max[j],
        )

    def range_extension(self, x_t, extensions):
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

    def __init__(self, parent: MondrianTreeLeafClassifier):
        super().__init__(parent)
        self.parent = parent

    def most_common_path(self) -> typing.Tuple[int, typing.Union["Leaf", "Branch"]]:
        raise NotImplementedError

    @property
    def repr_split(self):
        raise NotImplementedError
