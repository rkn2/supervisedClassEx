import mglearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = LinearRegression().fit(X_train, y_train)

'''The “slope” parameters (w), also called weights or coefficients, are stored in
the coef_ attribute, while the offset or intercept (b) is stored in the intercept_ attribute:
'''
print("lr.coef_:", lr.coef_)
print("lr.intercept_:", lr.intercept_)

'''FYI You might notice the strange-looking trailing underscore at the end
of coef_ and intercept_. scikit-learn always stores anything that is derived from the
training data in attributes that end with a trailing underscore. That is to separate them
from parameters that are set by the user. The intercept_ attribute is always a single float number, while the coef_ attribute is a
NumPy array with one entry per input feature. As we only have a single input feature in
the wave dataset, lr.coef_ only has a single entry.'''

# Let’s look at the training set and test set performance:

print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

'''An R2 of around 0.66 is not very good, but we can see that the scores on the training and
test sets are very close together. This means we are likely underfitting, not overfitting. For
this one-dimensional dataset, there is little danger of overfitting, as the model is very
simple (or restricted). However, with higher-dimensional datasets (meaning datasets with
a large number of features), linear models become more powerful, and there is a higher
chance of overfitting.'''


# so let's try a more complex dataset, boston housing dataset
'''this dataset has 506
samples and 104 derived features. First, we load the dataset and split it into a training and
a test set. Then we build the linear regression model as before:'''

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)


'''When comparing training set and test set scores, we find that we predict very accurately on
the training set, but the R2 on the test set is much worse:'''

print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

'''This discrepancy between performance on the training set and the test set is a clear sign of
overfitting, and therefore we should try to find a model that allows us to control
complexity. One of the most commonly used alternatives to standard linear regression
is ridge regression,'''