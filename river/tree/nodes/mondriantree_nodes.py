from river.tree.nodes.mondriantree_utils import *

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


class NodesClassifier(object):
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

    def __init__(
            self, n_features, n_classes, n_samples_increment, n_nodes, n_nodes_capacity
    ):
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
        init_nodes(self, n_features, n_samples_increment, n_nodes, n_nodes_capacity)
        init_nodes_classifier(self, n_classes)


def init_nodes_arrays(nodes, n_nodes_capacity, n_features):
    """Initializes the nodes arrays given their capacity

    Parameters
    ----------
    nodes : :obj:`NodesClassifier` or :obj:`NodesRegressor`
        Object to be initialized

    n_nodes_capacity : :obj:`int`
        Desired nodes capacity

    n_features : :obj:`int`
        Number of features used during training.
    """
    nodes.index = np.zeros(n_nodes_capacity, dtype=uint32)
    nodes.is_leaf = np.ones(n_nodes_capacity, dtype=np.bool)
    nodes.depth = np.zeros(n_nodes_capacity, dtype=uint8)
    nodes.n_samples = np.zeros(n_nodes_capacity, dtype=uint32)
    nodes.parent = np.zeros(n_nodes_capacity, dtype=uint32)
    nodes.left = np.zeros(n_nodes_capacity, dtype=uint32)
    nodes.right = np.zeros(n_nodes_capacity, dtype=uint32)
    nodes.feature = np.zeros(n_nodes_capacity, dtype=uint32)
    nodes.weight = np.zeros(n_nodes_capacity, dtype=float32)
    nodes.log_weight_tree = np.zeros(n_nodes_capacity, dtype=float32)
    nodes.threshold = np.zeros(n_nodes_capacity, dtype=float32)
    nodes.time = np.zeros(n_nodes_capacity, dtype=float32)
    nodes.memory_range_min = np.zeros((n_nodes_capacity, n_features), dtype=float32)
    nodes.memory_range_max = np.zeros((n_nodes_capacity, n_features), dtype=float32)


def init_nodes(nodes, n_features, n_samples_increment, n_nodes, n_nodes_capacity):
    """Initializes a `Nodes` instance.

    Parameters
    ----------
    nodes : :obj:`NodesClassifier` or :obj:`NodesRegressor`
        Object to be initialized

    n_features : :obj:`int`
        Number of features used during training.

    n_samples_increment : :obj:`int`
        The minimum amount of memory which is pre-allocated each time extra memory is
        required for new nodes.

    n_nodes : :obj:`int`
        Blabla

    n_nodes_capacity : :obj:`int`
        Initial required node capacity. If 0, we use 2 * n_samples_increment + 1,
        otherwise we use the given value (useful for serialization).
    """

    if n_nodes_capacity == 0:
        # One for root + and twice the number of samples
        n_nodes_capacity = 2 * n_samples_increment + 1
    nodes.n_samples_increment = n_samples_increment
    nodes.n_features = n_features
    nodes.n_nodes_capacity = n_nodes_capacity
    nodes.n_nodes = n_nodes
    # Initialize node attributes
    init_nodes_arrays(nodes, n_nodes_capacity, n_features)


def init_nodes_classifier(nodes, n_classes):
    """Initializes a `NodesClassifier` instance.

    Parameters
    ----------
    n_classes : :obj:`int`
        Number of expected classes in the labels.

    """
    nodes.counts = np.zeros((nodes.n_nodes_capacity, n_classes), dtype=np.uint32)
    nodes.n_classes = n_classes


def add_node_classifier(nodes, parent, time):
    """Adds a node with specified parent and creation time.

    Parameters
    ----------
    nodes : :obj:`Nodes`
        The collection of nodes.

    parent : :obj:`int`
        The index of the parent of the new node.

    time : :obj:`float`
        The creation time of the new node.

    Returns
    -------
    output : `int`
        Index of the new node.

    """
    if nodes.n_nodes >= nodes.n_nodes_capacity:
        # We don't have memory for this extra node, so let's create some
        reserve_nodes_classifier(nodes)

    return add_node(nodes, parent, time)


def add_node(nodes, parent, time):
    """Adds a node with specified parent and creation time. This functions assumes that
    a node has been already allocated by "child" functions `add_node_classifier` and
    `add_node_regressor`.

    Parameters
    ----------
    nodes : :obj:`Nodes`
        The collection of nodes.

    parent : :obj:`int`
        The index of the parent of the new node.

    time : :obj:`float`
        The creation time of the new node.

    Returns
    -------
    output : `int`
        Index of the new node.

    """
    node_index = nodes.n_nodes
    nodes.index[node_index] = node_index
    nodes.parent[node_index] = parent
    nodes.time[node_index] = time
    nodes.n_nodes += 1
    return nodes.n_nodes - 1


def reserve_nodes_classifier(nodes):
    """Reserves memory for classifier nodes.

    Parameters
    ----------
    nodes : :obj:`NodesClassifier`
        The collection of classifier nodes.

    """
    reserve_nodes(nodes)
    nodes.counts = resize_array(nodes.counts, nodes.n_nodes, nodes.n_nodes_capacity)


def reserve_nodes(nodes):
    """Reserves memory for nodes.

    Parameters
    ----------
    nodes : :obj:`Nodes`
        The collection of nodes.
    """
    n_nodes_capacity = nodes.n_nodes_capacity + 2 * nodes.n_samples_increment + 1
    n_nodes = nodes.n_nodes
    # TODO: why is this test useful ?
    if n_nodes_capacity > nodes.n_nodes_capacity:
        nodes.index = resize_array(nodes.index, n_nodes, n_nodes_capacity)
        # By default, a node is a leaf when newly created
        nodes.is_leaf = resize_array(nodes.is_leaf, n_nodes, n_nodes_capacity, fill=1)
        nodes.depth = resize_array(nodes.depth, n_nodes, n_nodes_capacity)
        nodes.n_samples = resize_array(nodes.n_samples, n_nodes, n_nodes_capacity)
        nodes.parent = resize_array(nodes.parent, n_nodes, n_nodes_capacity)
        nodes.left = resize_array(nodes.left, n_nodes, n_nodes_capacity)
        nodes.right = resize_array(nodes.right, n_nodes, n_nodes_capacity)
        nodes.feature = resize_array(nodes.feature, n_nodes, n_nodes_capacity)
        nodes.weight = resize_array(nodes.weight, n_nodes, n_nodes_capacity)
        nodes.log_weight_tree = resize_array(
            nodes.log_weight_tree, n_nodes, n_nodes_capacity
        )
        nodes.threshold = resize_array(nodes.threshold, n_nodes, n_nodes_capacity)
        nodes.time = resize_array(nodes.time, n_nodes, n_nodes_capacity)

        nodes.memory_range_min = resize_array(
            nodes.memory_range_min, n_nodes, n_nodes_capacity
        )
        nodes.memory_range_max = resize_array(
            nodes.memory_range_max, n_nodes, n_nodes_capacity
        )

    nodes.n_nodes_capacity = n_nodes_capacity


def node_classifier_score(tree, node, idx_class):
    """Computes the score of the node

    Parameters
    ----------
    tree : `TreeClassifier`
        The tree containing the node

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
    nodes = tree.nodes
    count = nodes.counts[node, idx_class]
    n_samples = nodes.n_samples[node]
    n_classes = tree.n_classes
    dirichlet = tree.dirichlet
    # We use the Jeffreys prior with dirichlet parameter
    return (count + dirichlet) / (n_samples + dirichlet * n_classes)


def node_classifier_predict(tree, idx_node, scores):
    # TODO: this is a bit silly ?... do everything at once
    for c in range(tree.n_classes):
        scores[c] = node_classifier_score(tree, idx_node, c)
    return scores


def node_classifier_loss(tree, node, idx_sample):
    c = tree.samples.labels[idx_sample]
    sc = node_classifier_score(tree, node, c)
    # TODO: benchmark different logarithms
    return -np.log(sc)


def node_classifier_update_weight(tree, idx_node, idx_sample):
    loss_t = node_classifier_loss(tree, idx_node, idx_sample)
    if tree.use_aggregation:
        tree.nodes.weight[idx_node] -= tree.step * loss_t
    return loss_t


def node_classifier_update_count(tree, idx_node, idx_sample):
    # TODO: Don't do it twice...
    c = tree.samples.labels[idx_sample]
    tree.nodes.counts[idx_node, c] += 1


def node_classifier_update_downwards(tree, idx_node, idx_sample, do_update_weight):
    x_t = tree.samples.features[idx_sample]
    nodes = tree.nodes
    n_features = tree.n_features
    memory_range_min = nodes.memory_range_min[idx_node]
    memory_range_max = nodes.memory_range_max[idx_node]
    # If it is the first sample, we copy the features vector into the min and
    # max range
    if nodes.n_samples[idx_node] == 0:
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
    nodes.n_samples[idx_node] += 1

    if do_update_weight:
        # TODO: Using x_t and y_t should be better...
        node_classifier_update_weight(tree, idx_node, idx_sample)

    node_classifier_update_count(tree, idx_node, idx_sample)


def node_classifier_is_dirac(tree, idx_node, y_t):
    c = y_t
    nodes = tree.nodes
    n_samples = nodes.n_samples[idx_node]
    count = nodes.counts[idx_node, c]
    return n_samples == count


def node_range(tree, idx_node, j):
    # TODO: do the version without memory...
    nodes = tree.nodes
    return (
        nodes.memory_range_min[idx_node, j],
        nodes.memory_range_max[idx_node, j],
    )


def node_compute_range_extension(tree, idx_node, x_t, extensions):
    extensions_sum = 0
    for j in range(tree.n_features):
        x_tj = x_t[j]
        feature_min_j, feature_max_j = node_range(tree, idx_node, j)
        if x_tj < feature_min_j:
            diff = feature_min_j - x_tj
        elif x_tj > feature_max_j:
            diff = x_tj - feature_max_j
        else:
            diff = 0
        extensions[j] = diff
        extensions_sum += diff
    return extensions_sum


def node_classifier_compute_split_time(tree, idx_node, idx_sample):
    samples = tree.samples
    nodes = tree.nodes
    y_t = samples.labels[idx_sample]
    #  Don't split if the node is pure: all labels are equal to the one of y_t
    if not tree.split_pure and node_classifier_is_dirac(tree, idx_node, y_t):
        return 0.0

    x_t = samples.features[idx_sample]
    extensions_sum = node_compute_range_extension(tree, idx_node, x_t, tree.intensities)
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


def copy_node(nodes, first, second):
    """Copies the node at index ``first`` into the node at index ``second``.

    Parameters
    ----------
    nodes : :obj:`Nodes`
        The collection of nodes.

    first : :obj:`int`
        The index of the node to be copied in ``second``.

    second : :obj:`int`
        The index of the node containing the copy of ``first``.

    """
    # We must NOT copy the index
    nodes.is_leaf[second] = nodes.is_leaf[first]
    nodes.depth[second] = nodes.depth[first]
    nodes.n_samples[second] = nodes.n_samples[first]
    nodes.parent[second] = nodes.parent[first]
    nodes.left[second] = nodes.left[first]
    nodes.right[second] = nodes.right[first]
    nodes.feature[second] = nodes.feature[first]
    nodes.weight[second] = nodes.weight[first]
    nodes.log_weight_tree[second] = nodes.log_weight_tree[first]
    nodes.threshold[second] = nodes.threshold[first]
    nodes.time[second] = nodes.time[first]
    nodes.memory_range_min[second, :] = nodes.memory_range_min[first, :]
    nodes.memory_range_max[second, :] = nodes.memory_range_max[first, :]


def copy_node_classifier(nodes, first, second):
    """Copies the node at index `first` into the node at index `second`.

    Parameters
    ----------
    nodes : :obj:`NodesClassifier`
        The collection of nodes

    first : :obj:`int`
        The index of the node to be copied in ``second``

    second : :obj:`int`
        The index of the node containing the copy of ``first``

    """
    copy_node(nodes, first, second)
    nodes.counts[second, :] = nodes.counts[first, :]


def node_classifier_split(
        tree, idx_node, split_time, threshold, feature, is_right_extension
):
    # Create the two splits
    nodes = tree.nodes
    left_new = add_node_classifier(nodes, idx_node, split_time)
    right_new = add_node_classifier(nodes, idx_node, split_time)
    if is_right_extension:
        # left_new is the same as idx_node, excepted for the parent, time and the
        #  fact that it's a leaf
        copy_node_classifier(nodes, idx_node, left_new)
        # so we need to put back the correct parent and time
        nodes.parent[left_new] = idx_node
        nodes.time[left_new] = split_time
        # right_new must have idx_node has parent
        nodes.parent[right_new] = idx_node
        nodes.time[right_new] = split_time
        # We must tell the old childs that they have a new parent, if the
        # current node is not a leaf
        if not nodes.is_leaf[idx_node]:
            left = nodes.left[idx_node]
            right = nodes.right[idx_node]
            nodes.parent[left] = left_new
            nodes.parent[right] = left_new
    else:
        copy_node_classifier(nodes, idx_node, right_new)
        nodes.parent[right_new] = idx_node
        nodes.time[right_new] = split_time
        nodes.parent[left_new] = idx_node
        nodes.time[left_new] = split_time
        if not nodes.is_leaf[idx_node]:
            left = nodes.left[idx_node]
            right = nodes.right[idx_node]
            nodes.parent[left] = right_new
            nodes.parent[right] = right_new

    nodes.feature[idx_node] = feature
    nodes.threshold[idx_node] = threshold
    nodes.left[idx_node] = left_new
    nodes.right[idx_node] = right_new
    nodes.is_leaf[idx_node] = False


def node_update_depth(tree, idx_node, depth):
    depth += 1
    nodes = tree.nodes
    nodes.depth[idx_node] = depth
    if nodes.is_leaf[idx_node]:
        return
    else:
        left = nodes.left[idx_node]
        right = nodes.right[idx_node]
        node_update_depth(tree, left, depth)
        node_update_depth(tree, right, depth)


def node_get_child(tree, idx_node, x_t):
    nodes = tree.nodes
    feature = nodes.feature[idx_node]
    threshold = nodes.threshold[idx_node]
    if x_t[feature] <= threshold:
        return nodes.left[idx_node]
    else:
        return nodes.right[idx_node]


def node_update_weight_tree(tree, idx_node):
    nodes = tree.nodes
    if nodes.is_leaf[idx_node]:
        nodes.log_weight_tree[idx_node] = nodes.weight[idx_node]
    else:
        left = nodes.left[idx_node]
        right = nodes.right[idx_node]
        weight = nodes.weight[idx_node]
        log_weight_tree = nodes.log_weight_tree
        log_weight_tree[idx_node] = log_sum_2_exp(
            weight, log_weight_tree[left] + log_weight_tree[right]
        )


def nodes_classifier_to_dict(nodes):
    d = {}
    for key, _ in spec_nodes_classifier:
        d[key] = getattr(nodes, key)
    return d


def dict_to_nodes(nodes, nodes_dict):
    nodes.index[:] = nodes_dict["index"]
    nodes.is_leaf[:] = nodes_dict["is_leaf"]
    nodes.depth[:] = nodes_dict["depth"]
    nodes.n_samples[:] = nodes_dict["n_samples"]
    nodes.parent[:] = nodes_dict["parent"]
    nodes.left[:] = nodes_dict["left"]
    nodes.right[:] = nodes_dict["right"]
    nodes.feature[:] = nodes_dict["feature"]
    nodes.weight[:] = nodes_dict["weight"]
    nodes.log_weight_tree[:] = nodes_dict["log_weight_tree"]
    nodes.threshold[:] = nodes_dict["threshold"]
    nodes.time[:] = nodes_dict["time"]
    nodes.memory_range_min[:] = nodes_dict["memory_range_min"]
    nodes.memory_range_max[:] = nodes_dict["memory_range_max"]


def dict_to_nodes_classifier(nodes, nodes_dict):
    dict_to_nodes(nodes, nodes_dict)
    nodes.counts[:] = nodes_dict["counts"]
