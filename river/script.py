import matplotlib.pyplot as plt
import numpy as np

from river.ensemble.amf import AMFClassifier

from river.datasets import Bananas


def plot_classes(X_test, y_test, X_train, y_train, title=None):
    fig, axs = plt.subplots()

    for c in np.unique(y_train):
        indexes_test = np.where(y_test == c)
        indexes_train = np.where(y_train == c)

        X_test_reduced = X_test[indexes_test]
        X_train_reduced = X_train[indexes_train]

        if len(indexes_test) == 0:
            print(f"No value found for class {c}")
            continue
        axs.scatter(X_test_reduced[:, 0], X_test_reduced[:, 1], label=str(c), color=f"C{c}")
        axs.scatter(X_train_reduced[:, 0], X_train_reduced[:, 1], color=f"C{c}", alpha=.1)

    axs.legend()
    if title is not None:
        fig.suptitle(title)


def plot_dataset(stream):
    X, y = [], []

    for x_t, y_t in stream:
        x_array = []
        for k in x_t:
            x_array.append(x_t[k])
        X.append(x_array)
        y.append(int(y_t))

    X, y = np.array(X), np.array(y)
    plot_classes(X, y, X, y, title="Banana Dataset")
    return X, y


stream = Bananas()

# Plotting the Banana dataset
X, y = plot_dataset(stream)

# Learning time

total_samples = 600  # set the amount of samples to iterate through
proportion_training = 0.5  # proportion of training samples

train_samples = proportion_training * total_samples  # set the amount of learning samples
test_samples = (1 - proportion_training) * total_samples  # set the amount of test samples

X_test, y_test = [], []  # arrays for the plot

amf = AMFClassifier(2, n_estimators=10, step=1.0, use_aggregation=True, dirichlet=0.1)

t = 0
for x_t, y_t in stream:
    if t < train_samples:
        amf.learn_one(x_t, int(y_t))  # learning sample (x_t, y_t)
    else:
        label = amf.predict_one(x_t)  # predicting one's class

        x_array = []
        for k in x_t:
            x_array.append(x_t[k])
        X_test.append(x_array)
        y_test.append(label)

    t += 1
    if t > total_samples:
        break

X_test, y_test = np.array(X_test), np.array(y_test)

plot_classes(X_test, y_test, X, y, title="Prediction with AMFClassifier")

plt.show()
