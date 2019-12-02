import knntest as kn
import numpy as np
from sklearn import datasets
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


iris = datasets.load_iris()

X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.7, random_state=1234)

k=4

ans = kn.infodist(X_train, y_train, k)
print(ans)
# print("accuracy of the new model", accuracy(y_test, ans))


default = KNeighborsClassifier(n_neighbors=4)
default.fit(X_train, y_train)
default_pred = default.predict(X_test)

print("Default Acc: " + str(accuracy(y_test, default_pred)))

print("Hamara Acc: " + str(accuracy(y_train, ans)))