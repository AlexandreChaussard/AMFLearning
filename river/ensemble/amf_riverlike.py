import numpy as np

from river.tree.mondrian_tree_classifier_riverlike import MondrianTreeClassifier
from river.utils.mondriantree_samples import *

from abc import ABC, abstractmethod

from river.base.classifier import MiniBatchClassifier
import pandas as pd


class AMFLearner(ABC):
    """Base class for Aggregated Mondrian Forest classifier and regressors for online
    learning.

    Note
    ----
    This class is not intended for end users but for development only.

    """

    def __init__(
            self,
            n_estimators,
            step,
            loss,
            use_aggregation,
            split_pure,
            n_samples_increment,
            random_state,
    ):
        """Instantiates a `AMFLearner` instance.

        Parameters
        ----------
        n_estimators : :obj:`int`
            The number of trees in the forest.

        step : :obj:`float`
            Step-size for the aggregation weights.

        loss : :obj:`str`
            The loss used for the computation of the aggregation weights.

        use_aggregation : :obj:`bool`
            Controls if aggregation is used in the trees. It is highly recommended to
            leave it as `True`.

        split_pure : :obj:`bool`
            Controls if nodes that contains only sample of the same class should be
            split ("pure" nodes). Default is `False`, namely pure nodes are not split,
            but `True` can be sometimes better.


        n_samples_increment : :obj:`int`
            Sets the minimum amount of memory which is pre-allocated each time extra
            memory is required for new samples and new nodes. Decreasing it can slow
            down training. If you know that each ``partial_fit`` will be called with
            approximately `n` samples, you can set n_samples_increment = `n` if `n` is
            larger than the default.

        random_state : :obj:`int` or :obj:`None`
            Controls the randomness involved in the trees.

        """
        # We will instantiate the numba class when data is passed to
        # `partial_fit`, since we need to know about `n_features` among others things
        self.no_python = None
        self._n_features = None
        self.n_estimators = n_estimators
        self.step = step
        self.loss = loss
        self.use_aggregation = use_aggregation
        self.split_pure = split_pure
        self.n_samples_increment = n_samples_increment
        self.random_state = random_state

    def partial_fit_helper(self, X, y):
        """Updates the classifier with the given batch of samples.

        Parameters
        ----------
        X : :obj:`np.ndarray`, shape=(n_samples, n_features)
            Input features matrix.

        y : :obj:`np.ndarray`
            Input labels vector.

        Returns
        -------
        output : :obj:`AMFClassifier`
            Updated instance of :obj:`AMFClassifier`

        """
        # First,ensure that X and y are C-contiguous and with float32 dtype

        n_samples, n_features = X.shape

        self._extra_y_test(y)
        # This is the first call to `partial_fit`, so we need to instantiate
        # the no python class
        if self.no_python is None:
            self._n_features = n_features
            self._instantiate_nopython_class()
        else:
            _, n_features = X.shape
            if n_features != self.n_features:
                raise ValueError(
                    "`partial_fit` was first called with n_features=%d while "
                    "n_features=%d in this call" % (self.n_features, n_features)
                )

        self._partial_fit(X, y)
        return self

    @abstractmethod
    def _partial_fit(self, X, y):
        pass

    @abstractmethod
    def _compute_weighted_depths(self, X):
        pass

    # TODO: such methods should be private
    def predict_helper(self, X):
        """Helper method for the predictions of the given features vectors. This is used
        in the ``predict`` and ``predict_proba`` methods.

        Parameters
        ----------
        X : :obj:`np.ndarray`, shape=(n_samples, n_features)
            Input features matrix to predict for.

        Returns
        -------
        output : :obj:`np.ndarray`
            Returns the predictions for the input features

        """

        n_samples, n_features = X.shape
        if not self.no_python:
            raise RuntimeError(
                "You must call `partial_fit` before asking for predictions"
            )
        else:
            if n_features != self.n_features:
                raise ValueError(
                    "`partial_fit` was called with n_features=%d while predictions are "
                    "asked with n_features=%d" % (self.n_features, n_features)
                )
        # TODO: this is useless for predictions ?!?
        predictions = self._compute_predictions(X)
        return predictions

    def weighted_depth_helper(self, X):

        n_samples, n_features = X.shape
        if not self.no_python:
            raise RuntimeError(
                "You must call `partial_fit` before asking for weighted depths"
            )
        else:
            if n_features != self.n_features:
                raise ValueError(
                    "`partial_fit` was called with n_features=%d while depths are "
                    "asked with n_features=%d" % (self.n_features, n_features)
                )
        weighted_depths = self._compute_weighted_depths(X)
        return weighted_depths

    def _compute_predictions(self, X):
        pass

    def _extra_y_test(self, y):
        pass

    def _instantiate_nopython_class(self):
        pass

    @property
    def n_features(self):
        """:obj:`int`: Number of features used during training."""
        return self._n_features

    @n_features.setter
    def n_features(self, val):
        raise ValueError("`n_features` is a readonly attribute")

    @property
    def n_estimators(self):
        """:obj:`int`: Number of trees in the forest."""
        return self._n_estimators

    @n_estimators.setter
    def n_estimators(self, val):
        if self.no_python:
            raise ValueError(
                "You cannot modify `n_estimators` after calling `partial_fit`"
            )
        else:
            if not isinstance(val, int):
                raise ValueError("`n_estimators` must be of type `int`")
            elif val < 1:
                raise ValueError("`n_estimators` must be >= 1")
            else:
                self._n_estimators = val

    @property
    def n_samples_increment(self):
        """:obj:`int`: Amount of memory pre-allocated each time extra memory is
        required."""
        return self._n_samples_increment

    @n_samples_increment.setter
    def n_samples_increment(self, val):
        if self.no_python:
            raise ValueError(
                "You cannot modify `n_samples_increment` after calling `partial_fit`"
            )
        else:
            if not isinstance(val, int):
                raise ValueError("`n_samples_increment` must be of type `int`")
            elif val < 1:
                raise ValueError("`n_samples_increment` must be >= 1")
            else:
                self._n_samples_increment = val

    @property
    def step(self):
        """:obj:`float`: Step-size for the aggregation weights."""
        return self._step

    @step.setter
    def step(self, val):
        if self.no_python:
            raise ValueError("You cannot modify `step` after calling `partial_fit`")
        else:
            if not isinstance(val, float):
                raise ValueError("`step` must be of type `float`")
            elif val <= 0:
                raise ValueError("`step` must be > 0")
            else:
                self._step = val

    @property
    def use_aggregation(self):
        """:obj:`bool`: Controls if aggregation is used in the trees."""
        return self._use_aggregation

    @use_aggregation.setter
    def use_aggregation(self, val):
        if self.no_python:
            raise ValueError(
                "You cannot modify `use_aggregation` after calling `partial_fit`"
            )
        else:
            if not isinstance(val, bool):
                raise ValueError("`use_aggregation` must be of type `bool`")
            else:
                self._use_aggregation = val

    @property
    def split_pure(self):
        """:obj:`bool`: Controls if nodes that contains only sample of the same class
        should be split."""
        return self._split_pure

    @split_pure.setter
    def split_pure(self, val):
        if self.no_python:
            raise ValueError(
                "You cannot modify `split_pure` after calling `partial_fit`"
            )
        else:
            if not isinstance(val, bool):
                raise ValueError("`split_pure` must be of type `bool`")
            else:
                self._split_pure = val

    @property
    def loss(self):
        """:obj:`str`: The loss used for the computation of the aggregation weights."""
        return "log"

    @loss.setter
    def loss(self, val):
        pass

    @property
    def random_state(self):
        """:obj:`int` or :obj:`None`: Controls the randomness involved in the trees."""
        if self._random_state == -1:
            return None
        else:
            return self._random_state

    @random_state.setter
    def random_state(self, val):
        if self.no_python:
            raise ValueError(
                "You cannot modify `random_state` after calling `partial_fit`"
            )
        else:
            if val is None:
                self._random_state = -1
            elif not isinstance(val, int):
                raise ValueError("`random_state` must be of type `int`")
            elif val < 0:
                raise ValueError("`random_state` must be >= 0")
            else:
                self._random_state = val

    def __repr__(self):
        pass


class AMFNoPython(object):
    def __init__(
            self,
            n_features,
            n_estimators,
            step,
            loss,
            use_aggregation,
            split_pure,
            n_samples_increment,
            samples: SamplesCollection,
            trees_iteration,
            trees_n_nodes,
            trees_n_nodes_capacity,
    ):
        self.n_features = n_features
        self.n_estimators = n_estimators
        self.step = step
        self.loss = loss
        self.use_aggregation = use_aggregation
        self.split_pure = split_pure
        self.n_samples_increment = n_samples_increment
        self.samples = samples


class AMFClassifierNoPython(AMFNoPython):
    def __init__(
            self,
            n_classes: int,
            n_features: int,
            n_estimators: int,
            step: float,
            loss: str,
            use_aggregation: bool,
            dirichlet: float,
            split_pure: bool,
            n_samples_increment: int,
            samples: SamplesCollection,
            trees_iteration,
            trees_n_nodes,
            trees_n_nodes_capacity,
    ):
        super().__init__(
            n_features,
            n_estimators,
            step,
            loss,
            use_aggregation,
            split_pure,
            n_samples_increment,
            samples,
            trees_iteration,
            trees_n_nodes,
            trees_n_nodes_capacity,
        )
        self.n_classes = n_classes
        self.dirichlet = dirichlet

        if trees_iteration.size == 0:
            self.iteration = 0
            # TODO: reflected lists will be replaced by typed list soon...
            iteration = 0
            trees = [
                MondrianTreeClassifier(
                    self.n_classes,
                    self.n_features,
                    self.step,
                    self.loss,
                    self.use_aggregation,
                    self.dirichlet,
                    self.split_pure,
                    self.samples,
                    iteration,
                )
                for _ in range(n_estimators)
            ]
            self.trees = trees
        else:
            trees = [
                MondrianTreeClassifier(
                    self.n_classes,
                    self.n_features,
                    self.step,
                    self.loss,
                    self.use_aggregation,
                    self.dirichlet,
                    self.split_pure,
                    self.samples,
                    trees_iteration[n_estimator],
                )
                for n_estimator in range(n_estimators)
            ]
            self.trees = trees


class AMFClassifier(AMFLearner, MiniBatchClassifier):
    """Aggregated Mondrian Forest classifier for online learning. This algorithm
    is truly online, in the sense that a single pass is performed, and that predictions
    can be produced anytime.

    Each node in a tree predicts according to the distribution of the labels
    it contains. This distribution is regularized using a "Jeffreys" prior
    with parameter ``dirichlet``. For each class with `count` labels in the
    node and `n_samples` samples in it, the prediction of a node is given by

        (count + dirichlet) / (n_samples + dirichlet * n_classes)

    The prediction for a sample is computed as the aggregated predictions of all the
    subtrees along the path leading to the leaf node containing the sample. The
    aggregation weights are exponential weights with learning rate ``step`` and loss
    ``loss`` when ``use_aggregation`` is ``True``.

    This computation is performed exactly thanks to a context tree weighting algorithm.
    More details can be found in the paper cited in references below.

    The final predictions are the average class probabilities predicted by each of the
    ``n_estimators`` trees in the forest.

    Note
    ----
    All the parameters of ``AMFClassifier`` become **read-only** after the first call
    to ``partial_fit``

    References
    ----------
    J. Mourtada, S. Gaiffas and E. Scornet, *AMF: Aggregated Mondrian Forests for Online Learning*, arXiv:1906.10529, 2019

    """

    def __init__(
            self,
            n_classes,
            n_estimators=10,
            step=1.0,
            loss="log",
            use_aggregation=True,
            dirichlet=None,
            split_pure=False,
            n_samples_increment=1024,
            random_state=None,
    ):
        """Instantiates a `AMFClassifier` instance.

        Parameters
        ----------
        n_classes : :obj:`int`
            Number of expected classes in the labels. This is required since we
            don't know the number of classes in advance in a online setting.

        n_estimators : :obj:`int`, default = 10
            The number of trees in the forest.

        step : :obj:`float`, default = 1
            Step-size for the aggregation weights. Default is 1 for classification with
            the log-loss, which is usually the best choice.

        loss : {"log"}, default = "log"
            The loss used for the computation of the aggregation weights. Only "log"
            is supported for now, namely the log-loss for multi-class
            classification.

        use_aggregation : :obj:`bool`, default = `True`
            Controls if aggregation is used in the trees. It is highly recommended to
            leave it as `True`.

        dirichlet : :obj:`float` or :obj:`None`, default = `None`
            Regularization level of the class frequencies used for predictions in each
            node. Default is dirichlet=0.5 for n_classes=2 and dirichlet=0.01 otherwise.

        split_pure : :obj:`bool`, default = `False`
            Controls if nodes that contains only sample of the same class should be
            split ("pure" nodes). Default is `False`, namely pure nodes are not split,
            but `True` can be sometimes better.

        n_samples_increment : :obj:`int`, default = 1024
            Sets the minimum amount of memory which is pre-allocated each time extra
            memory is required for new samples and new nodes. Decreasing it can slow
            down training. If you know that each ``partial_fit`` will be called with
            approximately `n` samples, you can set n_samples_increment = `n` if `n` is
            larger than the default.

        random_state : :obj:`int` or :obj:`None`, default = `None`
            Controls the randomness involved in the trees.
        """
        super().__init__(
            n_estimators=n_estimators,
            step=step,
            loss=loss,
            use_aggregation=use_aggregation,
            split_pure=split_pure,
            n_samples_increment=n_samples_increment,
            random_state=random_state
        )

        self.n_classes = n_classes
        if dirichlet is None:
            if self.n_classes == 2:
                self.dirichlet = 0.5
            else:
                self.dirichlet = 0.01
        else:
            self.dirichlet = dirichlet

        self._classes = set(range(n_classes))

    def _extra_y_test(self, y):
        if y.min() < 0:
            raise ValueError("All the values in `y` must be non-negative")
        y_max = y.max()
        if y_max not in self._classes:
            raise ValueError("n_classes=%d while y.max()=%d" % (self.n_classes, y_max))

    def _instantiate_nopython_class(self):
        trees_iteration = np.empty(0, dtype=np.uint32)
        trees_n_nodes = np.empty(0, dtype=np.uint32)
        trees_n_nodes_capacity = np.empty(0, dtype=np.uint32)
        n_samples = 0
        n_samples_capacity = 0
        samples = SamplesCollection(
            self.n_samples_increment, self.n_features, n_samples, n_samples_capacity
        )
        self.no_python = AMFClassifierNoPython(
            self.n_classes,
            self.n_features,
            self.n_estimators,
            self.step,
            self.loss,
            self.use_aggregation,
            self.dirichlet,
            self.split_pure,
            self.n_samples_increment,
            samples,
            trees_iteration,
            trees_n_nodes,
            trees_n_nodes_capacity,
        )

    def _partial_fit(self, X, y):
        print("Partial fit running...")
        n_samples_batch, n_features = X.shape
        # First, we save the new batch of data
        n_samples_before = self.no_python.samples.n_samples
        # Add the samples in the forest
        self.no_python.samples.add_samples(X, y)
        for i in range(n_samples_before, n_samples_before + n_samples_batch):
            print(f"Tree {i} is fitting...")
            # Then we fit all the trees using all new samples
            for tree in self.no_python.trees:
                tree.partial_fit(i)
            self.no_python.iteration += 1

    def _compute_weighted_depths(self, X):
        pass

    def learn_one(self, x: dict, y):
        # TODO: (River) Implement that function based on the `partial_fit`
        pass

    def learn_many(self, X: "pd.DataFrame", y: "pd.Series") -> "MiniBatchClassifier":
        # TODO: (River) Implement that function instead of `partial_fit`
        return self

    def partial_fit(self, X, y, classes=None):
        """Updates the classifier with the given batch of samples.

        Parameters
        ----------
        X : :obj:`np.ndarray`, shape=(n_samples, n_features)
            Input features matrix.

        y : :obj:`np.ndarray`
            Input labels vector.

        classes : :obj:`None`
            Must not be used, only here for backwards compatibility

        Returns
        -------
        output : :obj:`AMFClassifier`
            Updated instance of :obj:`AMFClassifier`

        """
        return AMFLearner.partial_fit_helper(self, X, y)

    def _predict_proba(self, X):

        n_samples_batch, _ = X.shape
        scores = np.zeros((n_samples_batch, self.no_python.n_classes), dtype="float32")

        scores_tree = np.empty(self.no_python.n_classes, float32)
        for i in range(n_samples_batch):
            scores_i = scores[i]
            x_i = X[i]
            # The prediction is simply the average of the predictions
            for tree in self.no_python.trees:
                tree.use_aggregation = self.no_python.use_aggregation
                scores_i += tree.predict_proba_one(x_i, scores_tree)
            scores_i /= self.no_python.n_estimators
        return scores

    def _compute_predictions(self, X):
        n_samples, n_features = X.shape
        scores = self._predict_proba(X)
        return scores

    def predict_proba_one(self, x: dict):
        # TODO: (River) Implementation that function based on `predict_proba` function
        pass

    def predict_proba_many(self, X: "pd.DataFrame") -> "pd.DataFrame":
        # TODO: (River) Implement that function using the `predict_proba` function
        pass

    def predict_proba(self, X):
        """Predicts the class probabilities for the given features vectors.

        Parameters
        ----------
        X : :obj:`np.ndarray`, shape=(n_samples, n_features)
            Input features matrix to predict for.

        Returns
        -------
        output : :obj:`np.ndarray`, shape=(n_samples, n_classes)
            Returns the predicted class probabilities for the input features

        """
        return AMFLearner.predict_helper(self, X)

    # TODO: put in AMFLearner and reorganize
    def predict_proba_tree(self, X, tree_index):
        """Predicts the class probabilities for the given features vectors using a
        single tree at given index ``tree``. Should be used only for debugging or
        visualisation purposes.

        Parameters
        ----------
        X : :obj:`np.ndarray`, shape=(n_samples, n_features)
            Input features matrix to predict for.

        tree_index : :obj:`int`
            Index of the tree, must be between 0 and ``n_estimators`` - 1

        Returns
        -------
        output : :obj:`np.ndarray`, shape=(n_samples, n_classes)
            Returns the predicted class probabilities for the input features

        """
        # TODO: unittests for this method
        if not self.no_python:
            raise RuntimeError(
                "You must call `partial_fit` before calling `predict_proba`"
            )
        else:
            n_samples, n_features = X.shape
            if n_features != self.n_features:
                raise ValueError(
                    "`partial_fit` was called with n_features=%d while `predict_proba` "
                    "received n_features=%d" % (self.n_features, n_features)
                )
            if not isinstance(tree_index, int):
                raise ValueError("`tree` must be of integer type")
            if tree_index < 0 or tree_index >= self.n_estimators:
                raise ValueError("`tree` must be between 0 and `n_estimators` - 1")

            n_samples_batch, _ = X.shape
            scores = np.empty((n_samples_batch, self.no_python.n_classes), dtype=float32)
            tree = self.no_python.trees[tree_index]
            for i in range(n_samples_batch):
                x_i = X[i]
                tree.use_aggregation = self.no_python.use_aggregation
                scores[i] += tree.predict_proba_one(x_i, scores[i])
            return scores

    @property
    def n_classes(self):
        """:obj:`int`: Number of expected classes in the labels."""
        return self._n_classes

    @n_classes.setter
    def n_classes(self, val):
        if self.no_python:
            raise ValueError(
                "You cannot modify `n_classes` after calling `partial_fit`"
            )
        else:
            if not isinstance(val, int):
                raise ValueError("`n_classes` must be of type `int`")
            elif val < 2:
                raise ValueError("`n_classes` must be >= 2")
            else:
                self._n_classes = val

    @property
    def dirichlet(self):
        """:obj:`float` or :obj:`None`: Regularization level of the class
        frequencies."""
        return self._dirichlet

    @dirichlet.setter
    def dirichlet(self, val):
        if self.no_python:
            raise ValueError(
                "You cannot modify `dirichlet` after calling `partial_fit`"
            )
        else:
            if not isinstance(val, float):
                raise ValueError("`dirichlet` must be of type `float`")
            elif val <= 0:
                raise ValueError("`dirichlet` must be > 0")
            else:
                self._dirichlet = val

    @property
    def loss(self):
        """:obj:`str`: The loss used for the computation of the aggregation weights."""
        return "log"

    @loss.setter
    def loss(self, val):
        pass

    def __repr__(self):
        pass
