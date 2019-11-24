from rnn import KNN
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


iris = datasets.load_digits()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=1234)

k = 5
clf = KNN(k=k)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

#print(X_train)
#print("aaaaaaaaaaaaaaaaaaaaaaa")
#print (X_test)


ypred = clf.predict(X_train)
#print("aaaaaaaaaaaaaaaaaaaaaa")
#print(ypred[50:])

shp = int(ypred.shape[0])
print(shp)
ans = abs(predictions[:shp] - ypred[:shp])

#print("custom KNN classification accuracy", accuracy(y_test, predictions))
#print(predictions.shape)
#print(y_test.shape)

#print("---")
print(ans)

ans = clf.infodist(X_test, ans)

print(ans)

loda = KNeighborsClassifier(n_neighbors = 4)
loda.fit(X_train, y_train)
lodapred = loda.predict(X_test)

print(lodapred)

print(Counter(lodapred).most_common(1))


