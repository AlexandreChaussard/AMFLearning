import pandas as pd
from numpy.random import uniform

from river.tree.mondrian_tree import MondrianTree

from river.tree.nodes.mondrian_tree_nodes import *
from river.utils.mondriantree_samples import *


class MondrianTreeClassifier(MondrianTree):
    """Mondrian Tree classifier.

    https://proceedings.neurips.cc/paper/2014/file/d1dc3a8270a6f9394f88847d7f0050cf-Paper.pdf

    Parameters
    ----------
        n_classes,
        n_features,
        step,
        loss,
        use_aggregation,
        dirichlet,
        split_pure,
        samples,
        iteration,
        n_nodes,
    """

    def __init__(
            self,
            n_classes: int,
            n_features: int,
            step: float,
            loss,
            use_aggregation: bool,
            dirichlet: float,
            split_pure: bool,
            iteration: int,
    ):

        super().__init__(
            n_features=n_features,
            step=step,
            loss=loss,
            use_aggregation=use_aggregation,
            split_pure=split_pure,
            iteration=iteration,
        )
        self.n_classes = n_classes
        self.dirichlet = dirichlet
        self.tree = MondrianTreeBranchClassifier(MondrianTreeLeafClassifier(None, self.n_features, 0.0, self.n_classes))

        # Training attributes
        # The current sample being proceeded
        self._x = None
        # The current label being proceeded
        self._y = None

    def score(self, node: MondrianTreeLeafClassifier):
        return node.score(self._y, self.dirichlet)

    def predict(self, node: MondrianTreeLeafClassifier, scores):
        return node.predict(self.dirichlet, scores)

    def loss(self, node: MondrianTreeLeafClassifier):
        return node.loss(self._y, self.dirichlet)

    def update_weight(self, node: MondrianTreeLeafClassifier):
        return node.update_weight(self._y, self.dirichlet, self.use_aggregation, self.step)

    def update_count(self, node: MondrianTreeLeafClassifier):
        return node.update_count(self._y)

    def update_downwards(self, node: MondrianTreeLeafClassifier, do_update_weight):
        return node.update_downwards(self._x, self._y, self.dirichlet, self.use_aggregation, self.step,
                                     do_update_weight)

    def compute_split_time(self, node: MondrianTreeLeafClassifier):
        #  Don't split if the node is pure: all labels are equal to the one of y_t
        if not self.split_pure and node.is_dirac(self._y):
            return 0.0

        extensions_sum = node.range_extension(self._x, self.intensities)
        # If x_t extends the current range of the node
        if extensions_sum > 0:
            # Sample an exponential with intensity = extensions_sum
            T = np.exp(1 / extensions_sum)
            time = node.time
            # Splitting time of the node (if splitting occurs)
            split_time = time + T
            # If the node is a leaf we must split it
            if node.is_leaf:
                return split_time
            # Otherwise we apply Mondrian process dark magic :)
            # 1. We get the creation time of the childs (left and right is the same)
            left = node.get_left()
            child_time = left.time
            # 2. We check if splitting time occurs before child creation time
            if split_time < child_time:
                return split_time

        return 0

    def split(
            self, node, split_time, threshold, feature, is_right_extension
    ):
        # Create the two splits

        def extend_child(parent, main_child, other):
            # Expending the node towards the main child chosen
            main_child.copy(parent)
            main_child.parent = parent
            main_child.time = split_time

            # other must have node has parent
            other.is_leaf = True

            # If the node previously had children, we have to update it
            if not parent.is_leaf:
                old_left = parent.get_left()
                old_right = parent.get_right()
                old_right.parent = main_child
                old_left.parent = main_child

        if is_right_extension:
            left = child_node = MondrianTreeLeafClassifier(node, self.n_features, split_time, self.n_classes)
            right = MondrianTreeLeafClassifier(node, self.n_features, split_time, self.n_classes)
            extend_child(node, left, right)

        else:
            right = MondrianTreeLeafClassifier(node, self.n_features, split_time, self.n_classes)
            left = MondrianTreeLeafClassifier(node, self.n_features, split_time, self.n_classes)
            extend_child(node, right, left)

        node.set_left(left)
        node.set_right(right)
        node.feature = feature
        node.threshold = threshold
        node.is_leaf = False

    def go_downwards(self):
        # We update the nodes along the path which leads to the leaf containing
        # x_t. For each node on the path, we consider the possibility of
        # splitting it, following the Mondrian process definition.
        current_node = self.tree.parent

        if self.iteration == 0:
            # If it's the first iteration, we just put x_t in the range of root
            self.update_downwards(current_node, False)
            return current_node
        else:
            while True:
                # If it's not the first iteration (otherwise the current node
                # is root with no range), we consider the possibility of a split
                split_time = self.compute_split_time(current_node)

                if split_time > 0:
                    # We split the current node: because the current node is a
                    # leaf, or because we add a new node along the path
                    # We normalize the range extensions to get probabilities
                    self.intensities /= self.intensities.sum()
                    # Sample the feature at random with a probability
                    # proportional to the range extensions
                    feature = sample_discrete(self.intensities)
                    x_tf = self._x[feature]
                    # Is it a right extension of the node ?
                    range_min, range_max = current_node.range(feature)
                    is_right_extension = x_tf > range_max
                    if is_right_extension:
                        threshold = uniform(range_max, x_tf)
                    else:
                        threshold = uniform(x_tf, range_min)

                    self.split(
                        current_node,
                        split_time,
                        threshold,
                        feature,
                        is_right_extension,
                    )

                    # Update the current node
                    self.update_downwards(current_node, True)

                    left = current_node.get_left()
                    right = current_node.get_right()
                    depth = current_node.depth

                    # Now, get the next node
                    if is_right_extension:
                        current_node = right
                    else:
                        current_node = left

                    left.update_depth(depth)
                    right.update_depth(depth)

                    # This is the leaf containing the sample point (we've just
                    # splitted the current node with the data point)
                    leaf = current_node
                    self.update_downwards(leaf, False)
                    return leaf
                else:
                    # There is no split, so we just update the node and go to
                    # the next one
                    self.update_downwards(current_node, True)
                    if current_node.is_leaf:
                        return current_node
                    else:
                        current_node = current_node.get_child(self._x)

    def go_upwards(self, leaf: MondrianTreeLeafClassifier):
        current_node = leaf
        if self.iteration >= 1:
            while True:
                current_node.update_weight_tree()
                if current_node.parent is None:
                    # We arrived at the root
                    break
                # Note that the root node is updated as well
                # We go up to the root in the tree
                current_node = current_node.parent

    def learn_one(self, x: dict, y: int):
        # setting current sample
        self._x = list(x.values())
        self._y = y

        # learning step
        leaf = self.go_downwards()
        if self.use_aggregation:
            self.go_upwards(leaf)

        # incrementing iteration
        self.iteration += 1
        return self

    @property
    def _multiclass(self):
        return True

    def find_leaf(self, x: dict) -> MondrianTreeLeafClassifier:
        # Find the leaf that contains the sample. Starts at the root.
        node = self.tree.parent

        is_leaf = False
        while not is_leaf:
            is_leaf = node.is_leaf
            if not is_leaf:
                feature_index = node.feature
                feature = list(x.keys())[feature_index]
                threshold = node.threshold
                if x[feature] <= threshold:
                    node = node.get_left()
                else:
                    node = node.get_right()
        return node

    def predict_proba_one(self, x: dict, init_scores=None) -> dict[int, float]:

        if init_scores is None:
            scores = {}  # [0] * self.n_classes
            for i in range(0, self.n_classes):
                scores[i] = 0
        else:
            scores = init_scores

        leaf = self.find_leaf(x)

        if not self.use_aggregation:
            scores = self.predict(leaf, scores)
            return scores

        current = leaf
        # Allocate once and for all
        pred_new = [0] * self.n_classes

        while True:
            # This test is useless ?
            if current.is_leaf:
                scores = self.predict(current, scores)
            else:
                weight = current.weight
                log_weight_tree = current.log_weight_tree
                w = np.exp(weight - log_weight_tree)
                # Get the predictions of the current node
                pred_new = self.predict(current, pred_new)
                for c in range(self.n_classes):
                    scores[c] = 0.5 * w * pred_new[c] + (1 - 0.5 * w) * scores[c]

            # Root must be updated as well
            if current.parent is None:
                break
            # And now we go up
            current = current.parent

        return scores

    def predict_one(self, x) -> int:
        scores = self.predict_proba_one(x)
        return max(range(len(scores)), key=lambda x: scores[x])
