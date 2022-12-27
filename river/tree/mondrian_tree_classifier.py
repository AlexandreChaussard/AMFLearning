import numpy as np
from numpy.random import uniform

from river.tree.mondrian_tree import MondrianTree
from river.tree.mondrian_tree import spec_tree

from river.tree.nodes.mondriantree_nodes import *
from river.tree.nodes.mondriantree_utils import *

spec_tree_classifier = spec_tree + [
    ("n_classes", uint32),
    ("dirichlet", float32),
    ("nodes", NodesClassifier),
]


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
        n_nodes_capacity
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
            samples: type(SamplesCollection),
            iteration: int,
            n_nodes: int,
            n_nodes_capacity: int,
    ):

        super().__init__(
            n_features=n_features,
            step=step,
            loss=loss,
            use_aggregation=use_aggregation,
            split_pure=split_pure,
            samples=samples,
            iteration=iteration,
            n_nodes=n_nodes,
            n_nodes_capacity=n_nodes_capacity
        )
        self.n_classes = n_classes
        self.dirichlet = dirichlet

        n_samples_increment = self.samples.n_samples_increment
        if n_nodes == 0:
            self.iteration = 0
            n_nodes = 0
            n_nodes_capacity = 0
            self.nodes = NodesClassifier(
                n_features, n_classes, n_samples_increment, n_nodes, n_nodes_capacity
            )
            add_node_classifier(self.nodes, 0, 0.0)
        else:
            self.iteration = iteration
            self.nodes = NodesClassifier(
                n_features, n_classes, n_samples_increment, n_nodes, n_nodes_capacity
            )

    def go_downwards(self, idx_sample):
        # We update the nodes along the path which leads to the leaf containing
        # x_t. For each node on the path, we consider the possibility of
        # splitting it, following the Mondrian process definition.
        # Index of the root is 0
        idx_current_node = 0
        x_t = self.samples.features[idx_sample]

        if self.iteration == 0:
            # If it's the first iteration, we just put x_t in the range of root
            node_classifier_update_downwards(self, idx_current_node, idx_sample, False)
            return idx_current_node
        else:
            while True:
                # If it's not the first iteration (otherwise the current node
                # is root with no range), we consider the possibility of a split
                split_time = node_classifier_compute_split_time(
                    self, idx_current_node, idx_sample
                )

                if split_time > 0:
                    # We split the current node: because the current node is a
                    # leaf, or because we add a new node along the path
                    # We normalize the range extensions to get probabilities
                    # TODO: faster than this ?
                    self.intensities /= self.intensities.sum()
                    # Sample the feature at random with a probability
                    # proportional to the range extensions
                    feature = sample_discrete(self.intensities)
                    x_tf = x_t[feature]
                    # Is it a right extension of the node ?
                    range_min, range_max = node_range(self, idx_current_node, feature)
                    is_right_extension = x_tf > range_max
                    if is_right_extension:
                        threshold = uniform(range_max, x_tf)
                    else:
                        threshold = uniform(x_tf, range_min)

                    node_classifier_split(
                        self,
                        idx_current_node,
                        split_time,
                        threshold,
                        feature,
                        is_right_extension,
                    )

                    # Update the current node
                    node_classifier_update_downwards(
                        self, idx_current_node, idx_sample, True
                    )

                    left = self.nodes.left[idx_current_node]
                    right = self.nodes.right[idx_current_node]
                    depth = self.nodes.depth[idx_current_node]

                    # Now, get the next node
                    if is_right_extension:
                        idx_current_node = right
                    else:
                        idx_current_node = left

                    node_update_depth(self, left, depth)
                    node_update_depth(self, right, depth)

                    # This is the leaf containing the sample point (we've just
                    # splitted the current node with the data point)
                    leaf = idx_current_node
                    node_classifier_update_downwards(self, leaf, idx_sample, False)
                    return leaf
                else:
                    # There is no split, so we just update the node and go to
                    # the next one
                    node_classifier_update_downwards(
                        self, idx_current_node, idx_sample, True
                    )
                    is_leaf = self.nodes.is_leaf[idx_current_node]
                    if is_leaf:
                        return idx_current_node
                    else:
                        idx_current_node = node_get_child(self, idx_current_node, x_t)

    def go_upwards(self, leaf):
        idx_current_node = leaf
        if self.iteration >= 1:
            while True:
                node_update_weight_tree(self, idx_current_node)
                if idx_current_node == 0:
                    # We arrived at the root
                    break
                # Note that the root node is updated as well
                # We go up to the root in the tree
                idx_current_node = self.nodes.parent[idx_current_node]

    def partial_fit(self, idx_sample):
        leaf = self.go_downwards(idx_sample)
        if self.use_aggregation:
            self.go_upwards(leaf)
        self.iteration += 1

    def learn_one(self, X, y):
        n_samples_batch, n_features = X.shape
        # First, we save the new batch of data
        n_samples_before = self.samples.n_samples
        # Add the samples in the forest
        add_samples(self.samples, X, y)
        for i in range(n_samples_before, n_samples_before + n_samples_batch):
            # Then we fit the tree using all new samples
            self.partial_fit(i)
            self.iteration += 1
        return self

    @property
    def _multiclass(self):
        return True

    def find_leaf(self, X):
        # Find the index of the leaf that contains the sample. Start at the root.
        # Index of the root is 0

        node = 0
        is_leaf = False
        while not is_leaf:
            is_leaf = self.nodes.is_leaf[node]
            if not is_leaf:
                feature = self.nodes.feature[node]
                threshold = self.nodes.threshold[node]
                if X[feature] <= threshold:
                    node = self.nodes.left[node]
                else:
                    node = self.nodes.right[node]
        return node

    def predict_proba_one(self, x, init_scores=None):

        if init_scores is None:
            scores = np.zeros(self.n_classes)
        else:
            scores = init_scores.copy()

        leaf = self.find_leaf(x)

        if not self.use_aggregation:
            node_classifier_predict(self, leaf, scores)
            return scores

        current = leaf
        # Allocate once and for all
        pred_new = np.empty(self.n_classes, float32)

        while True:
            # This test is useless ?
            if self.nodes.is_leaf[current]:
                node_classifier_predict(self, current, scores)
            else:
                weight = self.nodes.weight[current]
                log_weight_tree = self.nodes.log_weight_tree[current]
                w = np.exp(weight - log_weight_tree)
                # Get the predictions of the current node
                node_classifier_predict(self, current, pred_new)
                for c in range(self.n_classes):
                    scores[c] = 0.5 * w * pred_new[c] + (1 - 0.5 * w) * scores[c]

            # Root must be update as well
            if current == 0:
                break
            # And now we go up
            current = self.nodes.parent[current]

        return scores

    def predict_one(self, x):
        scores = self.predict_proba_one(x)
        return np.argmax(scores)

    def serialize(self):
        d = {}
        for key, dtype in spec_tree_classifier:
            if key == "nodes":
                nodes = nodes_classifier_to_dict(self.nodes)
                d["nodes"] = nodes
            elif key == "samples":
                # We do not save the samples here. There are saved in the forest
                # otherwise a copy is made for each tree in the pickle file
                pass
            else:
                d[key] = getattr(self, key)
        return d
