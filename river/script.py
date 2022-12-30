import matplotlib.pyplot as plt
import numpy as np

from river.ensemble.amf import AMFClassifier
from river.utils import data_conversion as conversion

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

total_samples = 500 # set the amount of samples to iterate through
proportion_training = 1  # proportion of training samples

train_samples = proportion_training * total_samples  # set the amount of learning samples
test_samples = (1 - proportion_training) * total_samples  # set the amount of test samples

X_test, y_pred, scores = [], [], []  # arrays for the plot

amf = AMFClassifier(2, n_estimators=10, step=1.0, use_aggregation=True, dirichlet=0.2)

t = 0
for x_t, y_t in stream:
    if t < train_samples:
        amf.learn_one(x_t, int(y_t))  # learning sample (x_t, y_t)
    else:
        score = list(amf.predict_proba_one(x_t).values())
        label = amf.predict_one(x_t)  # predicting one's class

        x_array = []
        for k in x_t:
            x_array.append(x_t[k])
        X_test.append(x_array)
        y_pred.append(label)
        scores.append(score)

    t += 1
    if t > total_samples:
        break

X_test, y_pred, scores = np.array(X_test), np.array(y_pred), np.array(scores)

plot_classes(X_test, y_pred, X, y, title="Prediction with AMFClassifier")


def plot_decision_areas(X, amf):

    fig, axs = plt.subplots()

    # define bounds of the domain
    min1, max1 = X[:, 0].min(), X[:, 0].max()
    min2, max2 = X[:, 1].min(), X[:, 1].max()
    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1, r2))

    # make predictions for the grid
    yhat = []
    for x_numpy in grid:
        x_t = conversion.numpy2dict(x_numpy)
        score = list(amf.predict_proba_one(x_t).values())
        yhat.append(score)

    yhat = np.array(yhat)
    # keep just the probabilities for class 0
    yhat = yhat[:, 0]
    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)
    # plot the grid of x, y and z values as a surface
    c = axs.contourf(xx, yy, zz, cmap='RdBu', vmin=0, vmax=1, levels=40)
    # add a legend, called a color bar
    fig.suptitle("Decision areas of AMF")
    plt.colorbar(c)


plot_decision_areas(X, amf)
plt.show()
