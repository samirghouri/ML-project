from re import KNN
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
    X, y, test_size=0.5, random_state=1234)

k = 5
clf = KNN(k=k)
clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)



ans = clf.infodist(X_train)
print(ans)
# print("accuracy of the new model", accuracy(y_test, ans))


loda = KNeighborsClassifier(n_neighbors=4)
loda.fit(X_train, y_train)
lodapred = loda.predict(X_test)

# print(lodapred)

print(Counter(lodapred).most_common(1))
