import typing

from abc import ABC, abstractmethod

from river.utils.mondrian_utils import *

from river.tree.base import Leaf
from river.tree.base import Branch


class MondrianTreeLeaf(Leaf, ABC):
    """In a Mondrian Tree, a node is defined by its index, its parent index (can be itself) and its creation time.
       The index is given by the tree, while the parent index and the creation time are variables of the node
    """

    def __init__(self, parent, n_features, time: float):
        super().__init__()

        # Generic Node attributes
        self.parent = parent
        self.time = time
        self.is_leaf = False
        self.depth = 0
        self.left = None
        self.right = None
        self.feature = 0
        self.weight = 0
        self.log_weight_tree = 0
        self.threshold = 0
        self.n_samples = 0
        self.n_features = n_features
        self.memory_range_min = np.zeros(n_features)
        self.memory_range_max = np.zeros(n_features)

    def copy(self, node):
        """Copies the node into the current one."""
        # We must NOT copy the index
        self.is_leaf = node.is_leaf
        self.depth = node.depth
        self.parent = node.parent
        self.left = node.left
        self.right = node.right
        self.feature = node.feature
        self.weight = node.weight
        self.log_weight_tree = node.log_weight_tree
        self.threshold = node.threshold
        self.time = node.time

    def set_depth(self, depth):
        self.depth = depth

        # if it's a leaf, no need to update the children too
        if self.is_leaf:
            return

        if self.left:
            self.left.set_depth(depth + 1)
        if self.right:
            self.right.set_depth(depth + 1)

    def update_weight_tree(self):

        if self.is_leaf:
            self.log_weight_tree = self.weight
        else:
            left = self.left
            left_weight = 0
            if left is not None:
                left_weight = left.log_weight_tree

            right = self.right
            right_weight = 0
            if right is not None:
                right_weight = right.log_weight_tree

            weight = self.weight

            self.log_weight_tree = log_sum_2_exp(
                weight, left_weight + right_weight
            )

    def get_child(self, x):
        test = self.left
        other = self.right

        # Testing if the children are defined
        if test is None:
            test = self.right
            other = self.left
        if test is None:
            return self

        if x[test.feature] <= test.threshold:
            return test
        else:
            return other

    @property
    def __repr__(self):
        return f"{self.parent}, {self.time}"


class MondrianTreeLeafClassifier(MondrianTreeLeaf):

    def __init__(self, parent, n_features, time: float, n_classes: int):
        super().__init__(parent, n_features, time)
        self.n_classes = n_classes
        self.counts = np.zeros(n_classes)

    def score(self, idx_class, dirichlet):
        """Computes the score of the node

        Parameters
        ----------
        dirichlet : `float`

        n_samples : `int`
            Number of samples in total

        idx_class : `int`
            Class index for which we want the score

        Returns
        -------
        output : `float32`
            The log-loss of the node

        Notes
        -----
        This uses Jeffreys prior with dirichlet parameter for smoothing
        """

        count = self.counts[idx_class]
        n_classes = self.n_classes
        # We use the Jeffreys prior with dirichlet parameter
        return (count + dirichlet) / (self.n_samples + dirichlet * n_classes)

    def predict(self, dirichlet, scores):
        for c in range(self.n_classes):
            scores[c] = self.score(c, dirichlet)
        return scores

    def loss(self, sample_class, dirichlet):
        sc = self.score(sample_class, dirichlet)
        return -np.log(sc)

    def update_weight(self, sample_class, dirichlet, use_aggregation, step):
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
        memory_range_min = self.memory_range_min
        memory_range_max = self.memory_range_max
        # If it is the first sample, we copy the features vector into the min and
        # max range
        if self.n_samples == 0:
            for j in range(self.n_features):
                x_tj = x_t[j]
                memory_range_min[j] = x_tj
                memory_range_max[j] = x_tj
        # Otherwise, we update the range
        else:
            for j in range(self.n_features):
                x_tj = x_t[j]
                if x_tj < memory_range_min[j]:
                    memory_range_min[j] = x_tj
                if x_tj > memory_range_max[j]:
                    memory_range_max[j] = x_tj

        # TODO: we should save the sample here and do a bunch of stuff about
        #  memorization
        # One more sample in the node
        self.n_samples += 1

        if do_update_weight:
            # TODO: Using x_t and y_t should be better...
            self.update_weight(sample_class, dirichlet, use_aggregation, step)

        self.update_count(sample_class)

    def range(self, j):
        # TODO: do the version without memory...
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
        super().__init__((parent.left, parent.right))
        self.parent = parent

    def next(self, x) -> typing.Union["Branch", "Leaf"]:

        left, right = self.parent.left, self.parent.right

        test = left
        other = right

        # Testing if the children are defined
        if test is None:
            test = right
            other = left
        if test is None:
            return self

        if x[test.feature] <= test.threshold:
            return test
        else:
            return other

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
