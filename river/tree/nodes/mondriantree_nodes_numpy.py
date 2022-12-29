from river.utils.mondrian_utils import *
from abc import ABC

spec_nodes = [
    # The index of the node in the tree
    ("index", np.array),
    # Is the node a leaf ?
    ("is_leaf", bool),
    # Depth of the node in the tree
    ("depth", np.array),
    # Number of samples in the node
    ("n_samples", np.array),
    # Index of the parent
    ("parent", np.array),
    # Index of the left child
    ("left", np.array),
    # Index of the right child
    ("right", np.array),
    # Index of the feature used for the split
    ("feature", np.array),
    # TODO: is it used ?
    # Logarithm of the aggregation weight for the node
    ("weight", np.array),
    # Logarithm of the aggregation weight for the sub-tree starting at this node
    ("log_weight_tree", np.array),
    # Threshold used for the split
    ("threshold", np.array),
    # Time of creation of the node
    ("time", np.array),
    # Minimum range of the data points in the node
    ("memory_range_min", np.array),
    # Maximum range of the data points in the node
    ("memory_range_max", np.array),
    # Number of features
    ("n_features", uint32),
    # Number of nodes actually used
    ("n_nodes", uint32),
    # For how many nodes do we allocate nodes in advance ?
    ("n_samples_increment", uint32),
    # Number of nodes currently allocated in memory
    ("n_nodes_capacity", uint32),
]

spec_nodes_classifier = spec_nodes + [
    # Counts the number of sample seen in each class
    ("counts", np.array),
    # Number of classes
    ("n_classes", uint32),
]


class Nodes(ABC):
    def __init__(
            self, n_features, n_samples_increment, n_nodes, n_nodes_capacity
    ):
        if n_nodes_capacity == 0:
            # One for root + and twice the number of samples
            n_nodes_capacity = 2 * n_samples_increment + 1
        self.n_samples_increment = n_samples_increment
        self.n_features = n_features
        self.n_nodes_capacity = n_nodes_capacity
        self.n_nodes = n_nodes

        # Initialize node attributes
        self.index = np.zeros(n_nodes_capacity, dtype=uint32)
        self.is_leaf = np.ones(n_nodes_capacity, dtype=bool)
        self.depth = np.zeros(n_nodes_capacity, dtype=uint8)
        self.n_samples = np.zeros(n_nodes_capacity, dtype=uint32)
        self.parent = np.zeros(n_nodes_capacity, dtype=uint32)
        self.left = np.zeros(n_nodes_capacity, dtype=uint32)
        self.right = np.zeros(n_nodes_capacity, dtype=uint32)
        self.feature = np.zeros(n_nodes_capacity, dtype=uint32)
        self.weight = np.zeros(n_nodes_capacity, dtype=float32)
        self.log_weight_tree = np.zeros(n_nodes_capacity, dtype=float32)
        self.threshold = np.zeros(n_nodes_capacity, dtype=float32)
        self.time = np.zeros(n_nodes_capacity, dtype=float32)
        self.memory_range_min = np.zeros((n_nodes_capacity, n_features), dtype=float32)
        self.memory_range_max = np.zeros((n_nodes_capacity, n_features), dtype=float32)

    def add_node(self, parent, time):
        """Adds a node with specified parent and creation time. This functions assumes that
        a node has been already allocated by "child" functions `add_node_classifier` and
        `add_node_regressor`.

        Parameters
        ----------
        parent : :obj:`int`
            The index of the parent of the new node.

        time : :obj:`float`
            The creation time of the new node.

        Returns
        -------
        output : `int`
            Index of the new node.

        """
        node_index = self.n_nodes
        self.index[node_index] = node_index
        self.parent[node_index] = parent
        self.time[node_index] = time
        self.n_nodes += 1
        return self.n_nodes - 1

    def reserve_nodes(self):
        """Reserves memory for nodes.
        """
        n_nodes_capacity = self.n_nodes_capacity + 2 * self.n_samples_increment + 1
        n_nodes = self.n_nodes
        # TODO: why is this test useful ?
        if n_nodes_capacity > self.n_nodes_capacity:
            self.index = resize_array(self.index, n_nodes, n_nodes_capacity)
            # By default, a node is a leaf when newly created
            self.is_leaf = resize_array(self.is_leaf, n_nodes, n_nodes_capacity, fill=1)
            self.depth = resize_array(self.depth, n_nodes, n_nodes_capacity)
            self.n_samples = resize_array(self.n_samples, n_nodes, n_nodes_capacity)
            self.parent = resize_array(self.parent, n_nodes, n_nodes_capacity)
            self.left = resize_array(self.left, n_nodes, n_nodes_capacity)
            self.right = resize_array(self.right, n_nodes, n_nodes_capacity)
            self.feature = resize_array(self.feature, n_nodes, n_nodes_capacity)
            self.weight = resize_array(self.weight, n_nodes, n_nodes_capacity)
            self.log_weight_tree = resize_array(
                self.log_weight_tree, n_nodes, n_nodes_capacity
            )
            self.threshold = resize_array(self.threshold, n_nodes, n_nodes_capacity)
            self.time = resize_array(self.time, n_nodes, n_nodes_capacity)

            self.memory_range_min = resize_array(
                self.memory_range_min, n_nodes, n_nodes_capacity
            )
            self.memory_range_max = resize_array(
                self.memory_range_max, n_nodes, n_nodes_capacity
            )

        self.n_nodes_capacity = n_nodes_capacity

    def copy_node(self, first, second):
        """Copies the node at index ``first`` into the node at index ``second``.

        Parameters
        ----------

        first : :obj:`int`
            The index of the node to be copied in ``second``.

        second : :obj:`int`
            The index of the node containing the copy of ``first``.

        """
        # We must NOT copy the index
        self.is_leaf[second] = self.is_leaf[first]
        self.depth[second] = self.depth[first]
        self.n_samples[second] = self.n_samples[first]
        self.parent[second] = self.parent[first]
        self.left[second] = self.left[first]
        self.right[second] = self.right[first]
        self.feature[second] = self.feature[first]
        self.weight[second] = self.weight[first]
        self.log_weight_tree[second] = self.log_weight_tree[first]
        self.threshold[second] = self.threshold[first]
        self.time[second] = self.time[first]
        self.memory_range_min[second, :] = self.memory_range_min[first, :]
        self.memory_range_max[second, :] = self.memory_range_max[first, :]

    def dict_to_nodes(self, nodes_dict):
        self.index[:] = nodes_dict["index"]
        self.is_leaf[:] = nodes_dict["is_leaf"]
        self.depth[:] = nodes_dict["depth"]
        self.n_samples[:] = nodes_dict["n_samples"]
        self.parent[:] = nodes_dict["parent"]
        self.left[:] = nodes_dict["left"]
        self.right[:] = nodes_dict["right"]
        self.feature[:] = nodes_dict["feature"]
        self.weight[:] = nodes_dict["weight"]
        self.log_weight_tree[:] = nodes_dict["log_weight_tree"]
        self.threshold[:] = nodes_dict["threshold"]
        self.time[:] = nodes_dict["time"]
        self.memory_range_min[:] = nodes_dict["memory_range_min"]
        self.memory_range_max[:] = nodes_dict["memory_range_max"]


class NodesClassifier(Nodes):
    """A collection of nodes for classification.

    Attributes
    ----------
    n_features : :obj:`int`
        Number of features used during training.

    n_nodes : :obj:`int`
        Number of nodes saved in the collection.

    n_samples_increment : :obj:`int`
        The minimum amount of memory which is pre-allocated each time extra memory is
        required for new nodes.

    n_nodes_capacity : :obj:`int`
        Number of nodes that can be currently saved in the object.

    """

    def __init__(self, n_features, n_classes, n_samples_increment, n_nodes, n_nodes_capacity):
        """Instantiates a `NodesClassifier` instance.

        Parameters
        ----------
        n_features : :obj:`int`
            Number of features used during training.

        n_classes : :obj:`int`
            Number of expected classes in the labels.

        n_samples_increment : :obj:`int`
            The minimum amount of memory which is pre-allocated each time extra memory
            is required for new nodes.
        """

        super().__init__(n_features, n_samples_increment, n_nodes, n_nodes_capacity)

        self.counts = np.zeros((self.n_nodes_capacity, n_classes), dtype=np.uint32)
        self.n_classes = n_classes

    def add_node_classifier(self, parent, time):
        """Adds a node with specified parent and creation time.

        Parameters
        ----------

        parent : :obj:`int`
            The index of the parent of the new node.

        time : :obj:`float`
            The creation time of the new node.

        Returns
        -------
        output : `int`
            Index of the new node.

        """
        if self.n_nodes >= self.n_nodes_capacity:
            # We don't have memory for this extra node, so let's create some
            self.reserve_nodes_classifier()

        return self.add_node(parent, time)

    def reserve_nodes_classifier(self):
        """Reserves memory for classifier nodes.
        """
        self.reserve_nodes()
        self.counts = resize_array(self.counts, self.n_nodes, self.n_nodes_capacity)

    def copy_node_classifier(self, first, second):
        """Copies the node at index `first` into the node at index `second`.

        Parameters
        ----------
        first : :obj:`int`
            The index of the node to be copied in ``second``

        second : :obj:`int`
            The index of the node containing the copy of ``first``

        """
        self.copy_node(first, second)
        self.counts[second, :] = self.counts[first, :]

spec_nodes_regressor = spec_nodes + [
    # Current mean of the labels in the node
    ("mean", float32[::1]),
]

class NodesRegressor(Nodes):
    """A collection of nodes for regression.
    Attributes
    ----------
    n_features : :obj:`int`
        Number of features used during training.
    n_nodes : :obj:`int`
        Number of nodes saved in the collection.
    n_samples_increment : :obj:`int`
        The minimum amount of memory which is pre-allocated each time extra memory is
        required for new nodes.
    n_nodes_capacity : :obj:`int`
        Number of nodes that can be currently saved in the object.
    """

    def __init__(self, n_features, n_samples_increment, n_nodes, n_nodes_capacity):
        """Instantiates a `NodesClassifier` instance.
        Parameters
        ----------
        n_features : :obj:`int`
            Number of features used during training.
        n_samples_increment : :obj:`int`
            The minimum amount of memory which is pre-allocated each time extra memory
            is required for new nodes.
        """
        super().__init__(n_features, n_samples_increment, n_nodes, n_nodes_capacity)
        self.mean = np.zeros(self.n_nodes_capacity, dtype=float32)

    def add_node_regressor(self, parent, time):
        """Adds a node with specified parent and creation time.
        Parameters
        ----------
        parent : :obj:`int`
            The index of the parent of the new node.
        time : :obj:`float`
            The creation time of the new node.
        Returns
        -------
        output : `int`
            Index of the new node.
        """
        if self.n_nodes >= self.n_nodes_capacity:
            # We don't have memory for this extra node, so let's create some
            self.reserve_nodes_regressor()

        return self.add_node(parent, time)

    def reserve_nodes_regressor(self):
        """Reserves memory for regressor nodes.
        Parameters
        ----------

        """
        self.reserve_nodes()
        self.mean = resize_array(self.mean, self.n_nodes, self.n_nodes_capacity)

    def copy_node_regressor(self, first, second):
        """Copies the node at index `first` into the node at index `second`.
        Parameters
        ----------
        first : :obj:`int`
            The index of the node to be copied in ``second``
        second : :obj:`int`
            The index of the node containing the copy of ``first``
        """
        self.copy_node(first, second)
        self.mean[second] = self.mean[first]










