from sklearn.linear_model import Lasso
import mglearn
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

# let's use the boston housing dataset again
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


lasso = Lasso().fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used:", np.sum(lasso.coef_ != 0))

'''As you can see, Lasso does quite badly, both on the training and the test set. This indicates
that we are underfitting, and we find that it used only 4 of the 104 features. Similarly
to Ridge, the Lasso also has a regularization parameter, alpha, that controls how strongly
coefficients are pushed toward zero. In the previous example, we used the default
of alpha=1.0. To reduce underfitting, let’s try decreasing alpha. When we do this, we also
need to increase the default setting of max_iter (the maximum number of iterations to
run):'''

# we increase the default setting of "max_iter",
# otherwise the model would warn us that we should increase max_iter.
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Number of features used:", np.sum(lasso001.coef_ != 0))

'''A lower alpha allowed us to fit a more complex model, which worked better on the training
and test data. The performance is slightly better than using Ridge, and we are using only 33
of the 104 features. This makes this model potentially easier to understand.'''

'''If we set alpha too low, however, we again remove the effect of regularization and end up
overfitting, with a result similar to LinearRegression:'''

lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Number of features used:", np.sum(lasso00001.coef_ != 0))

'''Again, we can plot the coefficients of the different models,'''

plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))
plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")

plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")

'''For alpha=1, we not only see that most of the coefficients are zero (which we already
knew), but that the remaining coefficients are also small in magnitude.
Decreasing alpha to 0.01, we obtain the solution shown as an upward pointing triangle,
which causes most features to be exactly zero. Using alpha=0.0001, we get a model that is
quite unregularized, with most coefficients nonzero and of large magnitude. For
comparison, the best Ridge solution is shown as circles. The Ridge model
with alpha=0.1 has similar predictive performance as the lasso model with alpha=0.01,
but using Ridge, all coefficients are nonzero.'''

'''In practice, ridge regression is usually the first choice between these two models. However,
if you have a large amount of features and expect only a few of them to be
important, Lasso might be a better choice. Similarly, if you would like to have a model that
is easy to interpret, Lasso will provide a model that is easier to understand, as it will select
only a subset of the input features. scikit-learn also provides the ElasticNet class, which
combines the penalties of Lasso and Ridge. In practice, this combination works best,
though at the price of having two parameters to adjust: one for the L1 regularization, and
one for the L2 regularization.'''