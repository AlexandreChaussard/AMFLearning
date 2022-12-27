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
        self.init_nodes()

    def init_nodes(self):
        n_samples_increment = self.samples.n_samples_increment
        if self.n_nodes == 0:
            self.iteration = 0
            n_nodes = 0
            n_nodes_capacity = 0
            self.nodes = NodesClassifier(
                self.n_features, self.n_classes, n_samples_increment, n_nodes, n_nodes_capacity
            )
            self.nodes.add_node_classifier(0, 0.0)
        else:
            # self.iteration is already initialized by default to the given value
            self.nodes = NodesClassifier(
                self.n_features, self.n_classes, n_samples_increment, self.n_nodes, self.n_nodes_capacity
            )

    def node_classifier_score(self, node, idx_class):
        """Computes the score of the node

        Parameters
        ----------
        node : `uint32`
            The index of the node in the tree

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
        count = self.nodes.counts[node, idx_class]
        n_samples = self.nodes.n_samples[node]
        n_classes = self.n_classes
        dirichlet = self.dirichlet
        # We use the Jeffreys prior with dirichlet parameter
        return (count + dirichlet) / (n_samples + dirichlet * n_classes)

    def node_classifier_predict(self, idx_node, scores):
        # TODO: this is a bit silly ?... do everything at once
        for c in range(self.n_classes):
            scores[c] = self.node_classifier_score(idx_node, c)
        return scores

    def node_classifier_loss(self, node, idx_sample):
        c = self.samples.labels[idx_sample]
        sc = self.node_classifier_score(node, c)
        # TODO: benchmark different logarithms
        return -np.log(sc)

    def node_classifier_update_weight(self, idx_node, idx_sample):
        loss_t = self.node_classifier_loss(idx_node, idx_sample)
        if self.use_aggregation:
            self.nodes.weight[idx_node] -= self.step * loss_t
        return loss_t

    def node_classifier_update_count(self, idx_node, idx_sample):
        # TODO: Don't do it twice...
        c = self.samples.labels[idx_sample]
        self.nodes.counts[idx_node, c] += 1

    def node_classifier_update_downwards(self, idx_node, idx_sample, do_update_weight):
        x_t = self.samples.features[idx_sample]
        n_features = self.n_features
        memory_range_min = self.nodes.memory_range_min[idx_node]
        memory_range_max = self.nodes.memory_range_max[idx_node]
        # If it is the first sample, we copy the features vector into the min and
        # max range
        if self.nodes.n_samples[idx_node] == 0:
            for j in range(n_features):
                x_tj = x_t[j]
                memory_range_min[j] = x_tj
                memory_range_max[j] = x_tj
        # Otherwise, we update the range
        else:
            for j in range(n_features):
                x_tj = x_t[j]
                if x_tj < memory_range_min[j]:
                    memory_range_min[j] = x_tj
                if x_tj > memory_range_max[j]:
                    memory_range_max[j] = x_tj

        # TODO: we should save the sample here and do a bunch of stuff about
        #  memorization
        # One more sample in the node
        self.nodes.n_samples[idx_node] += 1

        if do_update_weight:
            # TODO: Using x_t and y_t should be better...
            self.node_classifier_update_weight(idx_node, idx_sample)

        self.node_classifier_update_count(idx_node, idx_sample)

    def node_classifier_is_dirac(self, idx_node, y_t):
        c = y_t
        n_samples = self.nodes.n_samples[idx_node]
        count = self.nodes.counts[idx_node, c]
        return n_samples == count

    def node_classifier_compute_split_time(self, idx_node, idx_sample):
        samples = self.samples
        nodes = self.nodes
        y_t = samples.labels[idx_sample]
        #  Don't split if the node is pure: all labels are equal to the one of y_t
        if not self.split_pure and self.node_classifier_is_dirac(idx_node, y_t):
            return 0.0

        x_t = samples.features[idx_sample]
        extensions_sum = self.node_compute_range_extension(idx_node, x_t, self.intensities)
        # If x_t extends the current range of the node
        if extensions_sum > 0:
            # Sample an exponential with intensity = extensions_sum
            T = np.exp(1 / extensions_sum)
            time = nodes.time[idx_node]
            # Splitting time of the node (if splitting occurs)
            split_time = time + T
            # If the node is a leaf we must split it
            if nodes.is_leaf[idx_node]:
                return split_time
            # Otherwise we apply Mondrian process dark magic :)
            # 1. We get the creation time of the childs (left and right is the same)
            left = nodes.left[idx_node]
            child_time = nodes.time[left]
            # 2. We check if splitting time occurs before child creation time
            if split_time < child_time:
                return split_time

        return 0

    def node_classifier_split(
            self, idx_node, split_time, threshold, feature, is_right_extension
    ):
        # Create the two splits
        left_new = self.nodes.add_node_classifier(idx_node, split_time)
        right_new = self.nodes.add_node_classifier(idx_node, split_time)
        if is_right_extension:
            # left_new is the same as idx_node, excepted for the parent, time and the
            #  fact that it's a leaf
            self.nodes.copy_node_classifier(idx_node, left_new)
            # so we need to put back the correct parent and time
            self.nodes.parent[left_new] = idx_node
            self.nodes.time[left_new] = split_time
            # right_new must have idx_node has parent
            self.nodes.parent[right_new] = idx_node
            self.nodes.time[right_new] = split_time
            # We must tell the old childs that they have a new parent, if the
            # current node is not a leaf
            if not self.nodes.is_leaf[idx_node]:
                left = self.nodes.left[idx_node]
                right = self.nodes.right[idx_node]
                self.nodes.parent[left] = left_new
                self.nodes.parent[right] = left_new
        else:
            self.nodes.copy_node_classifier(idx_node, right_new)
            self.nodes.parent[right_new] = idx_node
            self.nodes.time[right_new] = split_time
            self.nodes.parent[left_new] = idx_node
            self.nodes.time[left_new] = split_time
            if not self.nodes.is_leaf[idx_node]:
                left = self.nodes.left[idx_node]
                right = self.nodes.right[idx_node]
                self.nodes.parent[left] = right_new
                self.nodes.parent[right] = right_new

        self.nodes.feature[idx_node] = feature
        self.nodes.threshold[idx_node] = threshold
        self.nodes.left[idx_node] = left_new
        self.nodes.right[idx_node] = right_new
        self.nodes.is_leaf[idx_node] = False

    def nodes_classifier_to_dict(self):
        d = {}
        for key, _ in spec_nodes_classifier:
            d[key] = getattr(self, key)
        return d

    def dict_to_nodes_classifier(self, nodes_dict):
        self.nodes.dict_to_nodes(nodes_dict)
        self.nodes.counts[:] = nodes_dict["counts"]

    def go_downwards(self, idx_sample):
        # We update the nodes along the path which leads to the leaf containing
        # x_t. For each node on the path, we consider the possibility of
        # splitting it, following the Mondrian process definition.
        # Index of the root is 0
        idx_current_node = 0
        x_t = self.samples.features[idx_sample]

        if self.iteration == 0:
            # If it's the first iteration, we just put x_t in the range of root
            self.node_classifier_update_downwards(idx_current_node, idx_sample, False)
            return idx_current_node
        else:
            while True:
                # If it's not the first iteration (otherwise the current node
                # is root with no range), we consider the possibility of a split
                split_time = self.node_classifier_compute_split_time(idx_current_node, idx_sample)

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
                    range_min, range_max = self.node_range(idx_current_node, feature)
                    is_right_extension = x_tf > range_max
                    if is_right_extension:
                        threshold = uniform(range_max, x_tf)
                    else:
                        threshold = uniform(x_tf, range_min)

                    self.node_classifier_split(
                        idx_current_node,
                        split_time,
                        threshold,
                        feature,
                        is_right_extension,
                    )

                    # Update the current node
                    self.node_classifier_update_downwards(idx_current_node, idx_sample, True)

                    left = self.nodes.left[idx_current_node]
                    right = self.nodes.right[idx_current_node]
                    depth = self.nodes.depth[idx_current_node]

                    # Now, get the next node
                    if is_right_extension:
                        idx_current_node = right
                    else:
                        idx_current_node = left

                    self.node_update_depth(left, depth)
                    self.node_update_depth(right, depth)

                    # This is the leaf containing the sample point (we've just
                    # splitted the current node with the data point)
                    leaf = idx_current_node
                    self.node_classifier_update_downwards(leaf, idx_sample, False)
                    return leaf
                else:
                    # There is no split, so we just update the node and go to
                    # the next one
                    self.node_classifier_update_downwards(idx_current_node, idx_sample, True)
                    is_leaf = self.nodes.is_leaf[idx_current_node]
                    if is_leaf:
                        return idx_current_node
                    else:
                        idx_current_node = self.node_get_child(idx_current_node, x_t)

    def go_upwards(self, leaf):
        idx_current_node = leaf
        if self.iteration >= 1:
            while True:
                self.node_update_weight_tree(idx_current_node)
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
            self.node_classifier_predict(leaf, scores)
            return scores

        current = leaf
        # Allocate once and for all
        pred_new = np.empty(self.n_classes, float32)

        while True:
            # This test is useless ?
            if self.nodes.is_leaf[current]:
                self.node_classifier_predict(current, scores)
            else:
                weight = self.nodes.weight[current]
                log_weight_tree = self.nodes.log_weight_tree[current]
                w = np.exp(weight - log_weight_tree)
                # Get the predictions of the current node
                self.node_classifier_predict(current, pred_new)
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
                nodes = self.nodes_classifier_to_dict()
                d["nodes"] = nodes
            elif key == "samples":
                # We do not save the samples here. There are saved in the forest
                # otherwise a copy is made for each tree in the pickle file
                pass
            else:
                d[key] = getattr(self, key)
        return d
