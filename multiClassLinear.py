import mglearn
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC

X, y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1", "Class 2"])

'''Now, we train a LinearSVC classifier on the dataset:'''

linear_svm = LinearSVC().fit(X, y)
print("Coefficient shape: ", linear_svm.coef_.shape)
print("Intercept shape: ", linear_svm.intercept_.shape)

'''We see that the shape of the coef_ is (3, 2), meaning that each row of coef_ contains the coefficient vector for 
one of the three classes and each column holds the coefficient value for a specific feature (there are two in this 
dataset). The intercept_ is now a one-dimensional array, storing the intercepts for each class. 

Let’s visualize the lines given by the three binary classifiers 
'''

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,
                                  mglearn.cm3.colors):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1',
            'Line class 2'], loc=(1.01, 0.3))

'''You can see that all the points belonging to class 0 in the training data are above the line corresponding to 
class 0, which means they are on the “class 0” side of this binary classifier. The points in class 0 are above the 
line corresponding to class 2, which means they are classified as “rest” by the binary classifier for class 2. The 
points belonging to class 0 are to the left of the line corresponding to class 1, which means the binary classifier 
for class 1 also classifies them as “rest.” Therefore, any point in this area will be classified as class 0 by the 
final classifier (the result of the classification confidence formula for classifier 0 is greater than zero, 
while it is smaller than zero for the other two classes). 

But what about the triangle in the middle of the plot? All three binary classifiers classify points there as “rest.” 
Which class would a point there be assigned to? The answer is the one with the highest value for the classification 
formula: the class of the closest line. 

'''

'''
The following example (Figure 2-21) shows the predictions for all regions of the 2D space:
'''

mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,
                                  mglearn.cm3.colors):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1',
            'Line class 2'], loc=(1.01, 0.3))
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

