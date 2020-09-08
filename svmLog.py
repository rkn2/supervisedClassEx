import mglearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

#forge data set (foundations of rebel group emergence)
X, y = mglearn.datasets.make_forge()

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
    ax=ax, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title(clf.__class__.__name__)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
axes[0].legend()

'''In this figure, we have the first feature of the forge dataset on the x-axis and the second
feature on the y-axis. We display the decision boundaries found
by LinearSVC and LogisticRegression respectively as straight lines, separating the area
classified as class 1 on the top from the area classified as class 0 on the bottom. In other
words, any new data point that lies above the black line will be classified as class 1 by the
respective classifier, while any point that lies below the black line will be classified as class
0.
The two models come up with similar decision boundaries. Note that both misclassify two
of the points. By default, both models apply an L2 regularization, in the same way
that Ridge does for regression.'''

'''For LogisticRegression and LinearSVC the trade-off parameter that determines the
strength of the regularization is called C, and higher values of C correspond
to less regularization. In other words, when you use a high value for the
parameter C, LogisticRegression and LinearSVC try to fit the training set as best as
possible, while with low values of the parameter C, the models put more emphasis on
finding a coefficient vector (w) that is close to zero.'''

'''There is another interesting aspect of how the parameter C acts. Using low values of C will
cause the algorithms to try to adjust to the “majority” of data points, while using a higher
value of C stresses the importance that each individual data point be classified correctly.
Here is an illustration using LinearSVM'''

mglearn.plots.plot_linear_svc_regularization()

'''On the lefthand side, we have a very small C corresponding to a lot of regularization. Most
of the points in class 0 are at the bottom, and most of the points in class 1 are at the top.
The strongly regularized model chooses a relatively horizontal line, misclassifying two
points. In the center plot, C is slightly higher, and the model focuses more on the two
misclassified samples, tilting the decision boundary. Finally, on the righthand side, the very
high value of C in the model tilts the decision boundary a lot, now correctly classifying all
points in class 0. One of the points in class 1 is still misclassified, as it is not possible to
correctly classify all points in this dataset using a straight line. The model illustrated on the
righthand side tries hard to correctly classify all points, but might not capture the overall
layout of the classes well. In other words, this model is likely overfitting.
Similarly to the case of regression, linear models for classification might seem very
restrictive in low-dimensional spaces, only allowing for decision boundaries that are
straight lines or planes. Again, in high dimensions, linear models for classification become
very powerful, and guarding against overfitting becomes increasingly important when
considering more features.'''

#lets look at a dataset with more features than x,y
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, stratify=cancer.target, random_state=42)

logreg = LogisticRegression().fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

'''The default value of C=1 provides quite good performance, with 95% accuracy on both the
training and the test set. But as training and test set performance are very close, it is likely
that we are underfitting. Let’s try to increase C to fit a more flexible model:'''

logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg100.score(X_test, y_test)))

'''Using C=100 results in higher training set accuracy, and also a slightly increased test set
accuracy, confirming our intuition that a more complex model should perform better.
We can also investigate what happens if we use an even more regularized model than the
default of C=1, by setting C=0.01:'''

logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg001.score(X_test, y_test)))

'''As expected, when moving from an
already underfit model, both training and test set accuracy decrease relative to the default
parameters.
Finally, let’s look at the coefficients learned by the models with the three different settings
of the regularization parameter C'''

plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")
plt.legend()

'''If we desire a more interpretable model, using L1 regularization might help, as it limits the
model to using only a few features. Here is the coefficient plot and classification accuracies
for L1 regularization'''

for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
    lr_l1 = LogisticRegression(C=C, penalty="l1", solver='liblinear').fit(X_train, y_train)
    print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
    C, lr_l1.score(X_train, y_train)))
    print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
    C, lr_l1.score(X_test, y_test)))
    plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))
    plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
    plt.hlines(0, 0, cancer.data.shape[1])
    plt.xlabel("Feature")
    plt.ylabel("Coefficient magnitude")
    plt.ylim(-5, 5)
    plt.legend(loc=3)

'''As you can see, there are many parallels between linear models for binary classification
and linear models for regression. As in regression, the main difference between the models
is the penalty parameter, which influences the regularization and whether the model will
use all available features or select only a subset.'''

