'''Ridge regression is implemented in linear_model.Ridge. Let’s see how well it does on the
extended Boston Housing dataset:'''

from sklearn.linear_model import Ridge
import mglearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
ridge = Ridge().fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

'''Recalling that the training score for the lin reg was 0.95, the training set score of Ridge 
is lower than for LinearRegression, while the test set score is higher. 
This is consistent with our expectation. With linear regression,
we were overfitting our data. Ridge is a more restricted model, so we are less likely to
overfit. A less complex model means worse performance on the training set, but better
generalization. As we are only interested in generalization performance, we should choose
the Ridge model over the LinearRegression model.'''

'''The Ridge model makes a trade-off between the simplicity of the model (near-zero
coefficients) and its performance on the training set. How much importance the model
places on simplicity versus training set performance can be specified by the user, using
the alpha parameter. In the previous attempt, we used the default parameter alpha=1.0.
There is no reason why this will give us the best trade-off, though. The optimum setting
of alpha depends on the particular dataset we are using. Increasing alpha forces
coefficients to move more toward zero, which decreases training set performance but
might help generalization. For example:'''

ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))

'''Decreasing alpha allows the coefficients to be less restricted'''

'''For very small values of alpha, coefficients are barely restricted at all, and we
end up with a model that resembles LinearRegression'''

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))

'''Here, alpha=0.1 seems to be working well. We could try decreasing alpha even more to
improve generalization.'''

'''We can also get a more qualitative insight into how the alpha parameter changes the model
by inspecting the coef_ attribute of models with different values of alpha. A
higher alpha means a more restricted model, so we expect the entries of coef_ to have
smaller magnitude for a high value of alpha than for a low value of alpha.'''

lr = LinearRegression().fit(X_train, y_train)

plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()

'''Here, the x-axis enumerates the entries of coef_: x=0 shows the coefficient associated with
the first feature, x=1 the coefficient associated with the second feature, and so on up
to x=100. The y-axis shows the numeric values of the corresponding values of the
coefficients. The main takeaway here is that for alpha=10, the coefficients are mostly
between around –3 and 3. The coefficients for the Ridge model with alpha=1, are
somewhat larger. The dots corresponding to alpha=0.1 have larger magnitude still, and many of the dots corresponding to linear regression without any regularization (which
would be alpha=0) are so large they are outside of the chart.'''

'''Another way to understand the influence of regularization is to fix a value of alpha but vary
the amount of training data available. We can subsample the Boston Housing
dataset and evaluated LinearRegression and Ridge(alpha=1) on subsets of increasing size
(plots that show model performance as a function of dataset size are called learning
curves):'''
plt.figure(0)
mglearn.plots.plot_ridge_n_samples()

'''As one would expect, the training score is higher than the test score for all dataset sizes, for
both ridge and linear regression. Because ridge is regularized, the training score of ridge is
lower than the training score for linear regression across the board. However, the test
score for ridge is better, particularly for small subsets of the data. For less than 400 data
points, linear regression is not able to learn anything. As more and more data becomes
available to the model, both models improve, and linear regression catches up with ridge in 
the end. The lesson here is that with enough training data, regularization becomes less
important, and given enough data, ridge and linear regression will have the same
performance (the fact that this happens here when using the full dataset is just by chance).
Another interesting aspect of taht figure is the decrease in training performance for linear
regression. If more data is added, it becomes harder for a model to overfit, or memorize the
data.'''