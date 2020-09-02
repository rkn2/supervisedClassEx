# import to conda environment
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np

X, y = mglearn.datasets.make_wave(n_samples=40)
# split the wave dataset into a training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# instantiate the model and set the number of neighbors to consider to 3
reg = KNeighborsRegressor(n_neighbors=3)
# fit the model using the training data and training targets
reg.fit(X_train, y_train)

#Now we can make predictions on the test set

print("Test set predictions:\n", reg.predict(X_test))

'''You can also evaluate the model using the score method, which for regressors returns
the R2 score. The R2 score, also known as the coefficient of determination, is a measure of
goodness of a prediction for a regression model, and yields a score that’s usually between 0
and 1. A value of 1 corresponds to a perfect prediction, and a value of 0 corresponds to a
constant model that just predicts the mean of the training set responses, y_train. The
formulation of R2 used here can even be negative, which can indicate anticorrelated
predictions.'''

print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))

# Here, the score is 0.83, which indicates a relatively good model fit.

'''For our one-dimensional dataset, we can see what the predictions look like for all possible
feature values. To do this, we create a test dataset consisting of many points
on the x-axis, which corresponds to the single feature: '''

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# create 1,000 data points, evenly spaced between -3 and 3
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    # make predictions using 1, 3, or 9 neighbors
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
    ax.set_title(
    "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
    n_neighbors, reg.score(X_train, y_train),
    reg.score(X_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target","Test data/target"], loc="best")

'''As we can see from the plot, using only a single neighbor, each point in the training set has
an obvious influence on the predictions, and the predicted values go through all of the data
points. This leads to a very unsteady prediction. Considering more neighbors leads to
smoother predictions, but these do not fit the training data as well.'''