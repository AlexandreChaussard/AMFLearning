import datetime as dt
import itertools
import time
import typing

from river import base, metrics, stream, utils

__all__ = ["progressive_val_score"]


def _progressive_validation(
    dataset: base.typing.Dataset,
    model,
    metric: metrics.base.Metric,
    checkpoints: typing.Iterator[int],
    moment: typing.Union[str, typing.Callable[[dict], dt.datetime]] = None,
    delay: typing.Union[str, int, dt.timedelta, typing.Callable] = None,
    measure_time=False,
    measure_memory=False,
):

    # Check that the model and the metric are in accordance
    if not metric.works_with(model):
        raise ValueError(f"{metric.__class__.__name__} metric is not compatible with {model}")

    # Determine if predict_one or predict_proba_one should be used in case of a classifier
    if utils.inspect.isanomalydetector(model):
        pred_func = model.score_one
    elif utils.inspect.isclassifier(model) and not metric.requires_labels:  # type: ignore
        pred_func = model.predict_proba_one
    else:
        pred_func = model.predict_one

    preds = {}

    next_checkpoint = next(checkpoints, None)
    n_total_answers = 0
    if measure_time:
        start = time.perf_counter()

    for i, x, y, *kwargs in stream.simulate_qa(dataset, moment, delay, copy=True):

        kwargs = kwargs[0] if kwargs else {}

        # Case 1: no ground truth, just make a prediction
        if y is None:
            preds[i] = pred_func(x=x, **kwargs)
            continue

        # Case 2: there's a ground truth, model and metric can be updated
        y_pred = preds.pop(i)

        # Update the metric
        if y_pred != {} and y_pred is not None:
            metric.update(y_true=y, y_pred=y_pred)

        # Update the model
        if model._supervised:
            model.learn_one(x=x, y=y, **kwargs)
        else:
            model.learn_one(x=x, **kwargs)

        # Yield current results
        n_total_answers += 1
        if n_total_answers == next_checkpoint:
            if isinstance(metric, metrics.base.Metrics):
                results = {m.__class__.__name__: m for m in metric}
            else:
                results = {metric.__class__.__name__: metric}
            results["Step"] = n_total_answers
            if measure_time:
                now = time.perf_counter()
                results["Time"] = dt.timedelta(seconds=now - start)
            if measure_memory:
                results["Memory"] = model._raw_memory_usage
            yield results
            next_checkpoint = next(checkpoints, None)


def iter_progressive_val_score(
    dataset: base.typing.Dataset,
    model,
    metric: metrics.base.Metric,
    moment: typing.Union[str, typing.Callable] = None,
    delay: typing.Union[str, int, dt.timedelta, typing.Callable] = None,
    step=1,
    measure_time=False,
    measure_memory=False,
) -> typing.Generator:
    """Evaluates the performance of a model on a streaming dataset and yields results.

    This does exactly the same as `evaluate.progressive_val_score`. The only difference is that
    this function returns an iterator, yielding results at every step. This can be useful if you
    want to have control over what you do with the results. For instance, you might want to plot
    the results.

    Parameters
    ----------
    dataset
        The stream of observations against which the model will be evaluated.
    model
        The model to evaluate.
    metric
        The metric used to evaluate the model's predictions.
    moment
        The attribute used for measuring time. If a callable is passed, then it is expected to take
        as input a `dict` of features. If `None`, then the observations are implicitly timestamped
        in the order in which they arrive.
    delay
        The amount to wait before revealing the target associated with each observation to the
        model. This value is expected to be able to sum with the `moment` value. For instance, if
        `moment` is a `datetime.date`, then `delay` is expected to be a `datetime.timedelta`. If a
        callable is passed, then it is expected to take as input a `dict` of features and the
        target. If a `str` is passed, then it will be used to access the relevant field from the
        features. If `None` is passed, then no delay will be used, which leads to doing standard
        online validation.
    step
        Iteration number at which to yield results. This only takes into account the
        predictions, and not the training steps.
    measure_time
        Whether or not to measure the elapsed time.
    measure_memory
        Whether or not to measure the memory usage of the model.

    Examples
    --------

    Take the following model:

    {'ROCAUC': ROCAUC: 89.80%, 'Step': 200}
    {'ROCAUC': ROCAUC: 92.09%, 'Step': 400}
    {'ROCAUC': ROCAUC: 93.13%, 'Step': 600}
    {'ROCAUC': ROCAUC: 93.99%, 'Step': 800}
    {'ROCAUC': ROCAUC: 94.74%, 'Step': 1000}
    {'ROCAUC': ROCAUC: 95.03%, 'Step': 1200}

    References
    ----------
    [^1]: [Beating the Hold-Out: Bounds for K-fold and Progressive Cross-Validation](http://hunch.net/~jl/projects/prediction_bounds/progressive_validation/coltfinal.pdf)
    [^2]: [Grzenda, M., Gomes, H.M. and Bifet, A., 2019. Delayed labelling evaluation for data streams. Data Mining and Knowledge Discovery, pp.1-30](https://link.springer.com/content/pdf/10.1007%2Fs10618-019-00654-y.pdf)

    """

    yield from _progressive_validation(
        dataset,
        model,
        metric,
        checkpoints=itertools.count(step, step) if step else iter([]),
        moment=moment,
        delay=delay,
        measure_time=measure_time,
        measure_memory=measure_memory,
    )


def progressive_val_score(
    dataset: base.typing.Dataset,
    model,
    metric: metrics.base.Metric,
    moment: typing.Union[str, typing.Callable] = None,
    delay: typing.Union[str, int, dt.timedelta, typing.Callable] = None,
    print_every=0,
    show_time=False,
    show_memory=False,
    **print_kwargs,
) -> metrics.base.Metric:
    """Evaluates the performance of a model on a streaming dataset.

    This method is the canonical way to evaluate a model's performance. When used correctly, it
    allows you to exactly assess how a model would have performed in a production scenario.

    `dataset` is converted into a stream of questions and answers. At each step the model is either
    asked to predict an observation, or is either updated. The target is only revealed to the model
    after a certain amount of time, which is determined by the `delay` parameter. Note that under
    the hood this uses the `stream.simulate_qa` function to go through the data in arrival order.

    By default, there is no delay, which means that the samples are processed one after the other.
    When there is no delay, this function essentially performs progressive validation. When there
    is a delay, then we refer to it as delayed progressive validation.

    It is recommended to use this method when you want to determine a model's performance on a
    dataset. In particular, it is advised to use the `delay` parameter in order to get a reliable
    assessment. Indeed, in a production scenario, it is often the case that ground truths are made
    available after a certain amount of time. By using this method, you can reproduce this scenario
    and therefore truthfully assess what would have been the performance of a model on a given
    dataset.

    Parameters
    ----------
    dataset
        The stream of observations against which the model will be evaluated.
    model
        The model to evaluate.
    metric
        The metric used to evaluate the model's predictions.
    moment
        The attribute used for measuring time. If a callable is passed, then it is expected to take
        as input a `dict` of features. If `None`, then the observations are implicitly timestamped
        in the order in which they arrive.
    delay
        The amount to wait before revealing the target associated with each observation to the
        model. This value is expected to be able to sum with the `moment` value. For instance, if
        `moment` is a `datetime.date`, then `delay` is expected to be a `datetime.timedelta`. If a
        callable is passed, then it is expected to take as input a `dict` of features and the
        target. If a `str` is passed, then it will be used to access the relevant field from the
        features. If `None` is passed, then no delay will be used, which leads to doing standard
        online validation.
    print_every
        Iteration number at which to print the current metric. This only takes into account the
        predictions, and not the training steps.
    show_time
        Whether or not to display the elapsed time.
    show_memory
        Whether or not to display the memory usage of the model.
    print_kwargs
        Extra keyword arguments are passed to the `print` function. For instance, this allows
        providing a `file` argument, which indicates where to output progress.

    Examples
    --------

    Take the following model:
    [200] ROCAUC: 89.80%
    [400] ROCAUC: 92.09%
    [600] ROCAUC: 93.13%
    [800] ROCAUC: 93.99%
    [1,000] ROCAUC: 94.74%
    [1,200] ROCAUC: 95.03%
    ROCAUC: 95.04%

    We haven't specified a delay, therefore this is strictly equivalent to the following piece
    of code:
    ROCAUC: 95.04%

    When `print_every` is specified, the current state is printed at regular intervals. Under
    the hood, Python's `print` method is being used. You can pass extra keyword arguments to
    modify its behavior. For instance, you may use the `file` argument if you want to log the
    progress to a file of your choice.

    >>> with open('progress.log', 'w') as f:
    ...     metric = evaluate.progressive_val_score(
    ...         model=model,
    ...         dataset=datasets.Phishing(),
    ...         metric=metrics.ROCAUC(),
    ...         print_every=200,
    ...         file=f
    ...     )

    >>> with open('progress.log') as f:
    ...     for line in f.read().splitlines():
    ...         print(line)
    [200] ROCAUC: 94.00%
    [400] ROCAUC: 94.70%
    [600] ROCAUC: 95.17%
    [800] ROCAUC: 95.42%
    [1,000] ROCAUC: 95.82%
    [1,200] ROCAUC: 96.00%

    Note that the performance is slightly better than above because we haven't used a fresh
    copy of the model. Instead, we've reused the existing model which has already done a full
    pass on the data.

    >>> import os; os.remove('progress.log')

    References
    ----------
    [^1]: [Beating the Hold-Out: Bounds for K-fold and Progressive Cross-Validation](http://hunch.net/~jl/projects/prediction_bounds/progressive_validation/coltfinal.pdf)
    [^2]: [Grzenda, M., Gomes, H.M. and Bifet, A., 2019. Delayed labelling evaluation for data streams. Data Mining and Knowledge Discovery, pp.1-30](https://link.springer.com/content/pdf/10.1007%2Fs10618-019-00654-y.pdf)

    """

    checkpoints = iter_progressive_val_score(
        dataset=dataset,
        model=model,
        metric=metric,
        moment=moment,
        delay=delay,
        step=print_every,
        measure_time=show_time,
        measure_memory=show_memory,
    )

    for checkpoint in checkpoints:

        msg = f"[{checkpoint['Step']:,d}] {metric}"
        if show_time:
            H, rem = divmod(checkpoint["Time"].seconds, 3600)
            M, S = divmod(rem, 60)
            msg += f" – {H:02d}:{M:02d}:{S:02d}"
        if show_memory:
            msg += f" – {utils.pretty.humanize_bytes(checkpoint['Memory'])}"
        print(msg, **print_kwargs)

    return metric
