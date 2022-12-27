from abc import abstractmethod

from river.tree.nodes.mondriantree_nodes import *
from river.utils.mondriantree_samples import SamplesCollection

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

        # nodes should be initialized with "init_nodes" that has to be overriden in the herited class
        self.nodes = None

    @abstractmethod
    def init_nodes(self):
        """
        initialize the nodes
        """
        pass

    def node_compute_range_extension(self, idx_node, x_t, extensions):
        extensions_sum = 0
        for j in range(self.n_features):
            x_tj = x_t[j]
            feature_min_j, feature_max_j = self.node_range(idx_node, j)
            if x_tj < feature_min_j:
                diff = feature_min_j - x_tj
            elif x_tj > feature_max_j:
                diff = x_tj - feature_max_j
            else:
                diff = 0
            extensions[j] = diff
            extensions_sum += diff
        return extensions_sum

    def node_update_depth(self, idx_node, depth):
        depth += 1
        self.nodes.depth[idx_node] = depth
        if self.nodes.is_leaf[idx_node]:
            return
        else:
            left = self.nodes.left[idx_node]
            right = self.nodes.right[idx_node]
            self.node_update_depth(left, depth)
            self.node_update_depth(right, depth)

    def node_get_child(self, idx_node, x_t):
        feature = self.nodes.feature[idx_node]
        threshold = self.nodes.threshold[idx_node]
        if x_t[feature] <= threshold:
            return self.nodes.left[idx_node]
        else:
            return self.nodes.right[idx_node]

    def node_update_weight_tree(self, idx_node):
        if self.nodes.is_leaf[idx_node]:
            self.nodes.log_weight_tree[idx_node] = self.nodes.weight[idx_node]
        else:
            left = self.nodes.left[idx_node]
            right = self.nodes.right[idx_node]
            weight = self.nodes.weight[idx_node]
            log_weight_tree = self.nodes.log_weight_tree
            log_weight_tree[idx_node] = log_sum_2_exp(
                weight, log_weight_tree[left] + log_weight_tree[right]
            )

    def node_range(self, idx_node, j):
        # TODO: do the version without memory...
        return (
            self.nodes.memory_range_min[idx_node, j],
            self.nodes.memory_range_max[idx_node, j],
        )
