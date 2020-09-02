# in terminal
# pip install sklearn
# pip install mg learn

# import to conda environment
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import mglearn
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# load dataset
X, y = mglearn.datasets.make_forge()

# instantiate the class
# test_train_split is built into sklearn. Put your cursor in the name and then press F1 for details
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# set number of neighbors to use
clf = KNeighborsClassifier(n_neighbors=3)

# fit classifier using the training set
clf.fit(X_train, y_train)

'''to make predictions on the test data, we can call the predict method. for each data point in the test set,
# this computes its nearest neighbors in the training set and finds the most common class among these:'''
print("Test set predictions:", clf.predict(X_test))

#to evaluate how well our model generalizes, we can use the score method with the test data together with the test labels:
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))
# We see that our model is about 86% accurate, meaning the model predicted the class
# correctly for 86% of the samples in the test dataset.

'''For two-dimensional datasets, we can also illustrate the prediction for all possible test
points in the xy-plane. We color the plane according to the class that would be assigned to a
point in this region. This lets us view the decision boundary, which is the divide between
where the algorithm assigns class 0 versus where it assigns class 1. The following code
produces the visualizations of the decision boundaries for one, three, and nine neighbors'''

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
    # the fit method returns the object self, so we can instantiate
    # and fit in one line
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax,
    alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)

'''As you can see on the left in the figure, using a single neighbor results in a decision
boundary that follows the training data closely. Considering more and more neighbors
leads to a smoother decision boundary. A smoother boundary corresponds to a simpler
model. In other words, using few neighbors corresponds to high model complexity (as
shown on the right side), and using many neighbors corresponds to low
model complexity (as shown on the left side). If you consider the extreme case
where the number of neighbors is the number of all data points in the training set, each test
point would have exactly the same neighbors (all training points) and all predictions would
be the same: the class that is most frequent in the training set.'''

# go back to slides

