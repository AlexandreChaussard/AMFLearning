from numpy.random import uniform

from river.tree.mondrian_tree_riverlike import MondrianTree

from river.tree.nodes.mondrian_tree_nodes_riverlike import *
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
            samples: type(SamplesCollection),
            iteration: int,
    ):

        super().__init__(
            n_features=n_features,
            step=step,
            loss=loss,
            use_aggregation=use_aggregation,
            split_pure=split_pure,
            samples=samples,
            iteration=iteration,
        )
        self.n_classes = n_classes
        self.dirichlet = dirichlet
        self.tree = MondrianTreeBranchClassifier(MondrianTreeLeafClassifier(None, self.n_features, 0.0, self.n_classes))

    def score(self, node: MondrianTreeLeafClassifier, indx_class: int):
        return node.score(indx_class, self.dirichlet)

    def predict(self, node: MondrianTreeLeafClassifier, scores):
        return node.predict(self.dirichlet, scores)

    def loss(self, node: MondrianTreeLeafClassifier, indx_sample: int):
        c = self.samples.labels[indx_sample]
        return node.loss(c)

    def update_weight(self, node: MondrianTreeLeafClassifier, indx_sample: int):
        c = self.samples.labels[indx_sample]
        return node.update_weight(c, self.dirichlet, self.use_aggregation, self.step)

    def update_count(self, node: MondrianTreeLeafClassifier, indx_sample: int):
        c = self.samples.labels[indx_sample]
        return node.update_count(c)

    def update_downwards(self, node: MondrianTreeLeafClassifier, indx_sample, do_update_weight):
        x_t = self.samples.features[indx_sample]
        c = self.samples.labels[indx_sample]
        return node.update_downwards(x_t, c, self.dirichlet, self.use_aggregation, self.step, do_update_weight)

    def compute_split_time(self, node: MondrianTreeLeafClassifier, idx_sample):
        y_t = self.samples.labels[idx_sample]

        #  Don't split if the node is pure: all labels are equal to the one of y_t
        if not self.split_pure and node.is_dirac(y_t):
            return 0.0

        x_t = self.samples.features[idx_sample]
        extensions_sum = node.range_extension(x_t, self.intensities)
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

        if is_right_extension:
            # left is the same as node, excepted for the parent, time and the
            # fact that it's a leaf
            left = MondrianTreeLeafClassifier(node, self.n_features, split_time, self.n_classes)
            left.copy(node)
            left.parent = node
            left.time = split_time

            # right_new must have node has parent
            right = MondrianTreeLeafClassifier(node, self.n_features, split_time, self.n_classes)
            right.is_leaf = True

            # If the node previously had children, we have to update it
            if not node.is_leaf:
                old_left = node.get_left()
                old_right = node.get_right()
                old_right.parent = left
                old_left.parent = left

        else:
            right = MondrianTreeLeafClassifier(node, self.n_features, split_time, self.n_classes)
            right.copy(node)
            right.parent = node
            right.time = split_time

            left = MondrianTreeLeafClassifier(node, self.n_features, split_time, self.n_classes)
            left.is_leaf = True

            if not node.is_leaf:
                old_left = node.get_left()
                old_right = node.get_right()
                old_right.parent = right
                old_left.parent = right

        node.set_left(left)
        node.set_right(right)
        node.feature = feature
        node.threshold = threshold
        node.is_leaf = False

    def go_downwards(self, idx_sample):
        # We update the nodes along the path which leads to the leaf containing
        # x_t. For each node on the path, we consider the possibility of
        # splitting it, following the Mondrian process definition.
        print(f"[go downwards] - with sample {idx_sample}")
        current_node = self.tree.parent
        x_t = self.samples.features[idx_sample]

        if self.iteration == 0:
            # If it's the first iteration, we just put x_t in the range of root
            print("  * first iteration: update downwards")
            self.update_downwards(current_node, idx_sample, False)
            print("  * returning current node")
            return current_node
        else:
            print("  * entering loop")
            while True:
                # If it's not the first iteration (otherwise the current node
                # is root with no range), we consider the possibility of a split

                print("      - computing split time...")
                split_time = self.compute_split_time(current_node, idx_sample)

                if split_time > 0:
                    print("      - Split time is positive")
                    # We split the current node: because the current node is a
                    # leaf, or because we add a new node along the path
                    # We normalize the range extensions to get probabilities
                    # TODO: faster than this ?
                    print("      - computing intensities")
                    self.intensities /= self.intensities.sum()
                    # Sample the feature at random with a probability
                    # proportional to the range extensions
                    print("      - Sampling feature")
                    feature = sample_discrete(self.intensities)
                    x_tf = x_t[feature]
                    # Is it a right extension of the node ?
                    print("      - Computing range extension")
                    range_min, range_max = current_node.range(feature)
                    is_right_extension = x_tf > range_max
                    print("      - Computing threshold")
                    if is_right_extension:
                        threshold = uniform(range_max, x_tf)
                    else:
                        threshold = uniform(x_tf, range_min)

                    print("      - Splitting...")
                    self.split(
                        current_node,
                        split_time,
                        threshold,
                        feature,
                        is_right_extension,
                    )

                    print("      - Updating downards")
                    # Update the current node
                    self.update_downwards(current_node, idx_sample, True)

                    print("      - Fetching children nodes")
                    left = current_node.get_left()
                    right = current_node.get_right()
                    depth = current_node.depth

                    # Now, get the next node
                    if is_right_extension:
                        current_node = right
                    else:
                        current_node = left

                    print("      - Updating depth")
                    left.set_depth(depth)
                    right.set_depth(depth)

                    # This is the leaf containing the sample point (we've just
                    # splitted the current node with the data point)
                    print("      - Updating downards")
                    leaf = current_node
                    self.update_downwards(leaf, idx_sample, False)
                    return leaf
                else:
                    print("      * no split")
                    # There is no split, so we just update the node and go to
                    # the next one
                    self.update_downwards(current_node, idx_sample, True)
                    if current_node.is_leaf:
                        print("      * returning current node")
                        return current_node
                    else:
                        print("      * moving on child node")
                        current_node = current_node.get_child(x_t)

    def go_upwards(self, leaf: MondrianTreeLeafClassifier):
        print(f"[go upwards] running with {self.iteration} iteration...")
        current_node = leaf
        if self.iteration >= 1:
            while True:
                print("  * updating weight tree at node...")
                current_node.update_weight_tree()
                if current_node.parent is None:
                    # We arrived at the root
                    break
                # Note that the root node is updated as well
                # We go up to the root in the tree
                print(f"  * node still has a parent, moving towards parent: current depth is {current_node.depth}")
                current_node = current_node.parent

    def partial_fit(self, idx_sample):
        print(f"[partial fit] with sample {idx_sample}")
        print("[partial fit] - go downwards")
        leaf = self.go_downwards(idx_sample)
        if self.use_aggregation:
            print("[partial fit] - go upwards")
            self.go_upwards(leaf)
        self.iteration += 1

    def learn_many(self, X, y):
        n_samples_batch, n_features = X.shape
        # First, we save the new batch of data
        n_samples_before = self.samples.n_samples
        # Add the samples in the forest
        self.samples.add_samples(X, y)
        for i in range(n_samples_before, n_samples_before + n_samples_batch):
            # Then we fit the tree using all new samples
            self.partial_fit(i)
            self.iteration += 1
        return self

    @property
    def _multiclass(self):
        return True

    def find_leaf(self, X):
        # Find the leaf that contains the sample. Start at the root.
        node = self.tree.parent

        is_leaf = False
        while not is_leaf:
            is_leaf = node.is_leaf
            if not is_leaf:
                feature = node.feature
                threshold = node.threshold
                if X[feature] <= threshold:
                    node = node.get_left()
                else:
                    node = node.get_right()
        return node

    def predict_proba_one(self, x, init_scores=None):

        if init_scores is None:
            scores = np.zeros(self.n_classes)
        else:
            scores = init_scores.copy()

        leaf = self.find_leaf(x)

        if not self.use_aggregation:
            scores = self.predict(leaf, scores)
            return scores

        current = leaf
        # Allocate once and for all
        pred_new = np.empty(self.n_classes, float32)

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

    def predict_one(self, x):
        scores = self.predict_proba_one(x)
        return np.argmax(scores)
