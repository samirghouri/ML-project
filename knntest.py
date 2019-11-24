import numpy as np
from rnn import KNN
from sklearn import datasets
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

#Load Dataset
iris = datasets.load_wine()
#iris = datasets.load_iris()
#iris = datasets.load_breast_cancer()
X, y = iris.data, iris.target

#Test Train Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=1234)

#No. of neighbours to search (k)
k = 5

#Classification
clf = KNN(k=k)
clf.fit(X_train, y_train)

#Prediction of Test set
predictions = clf.predict(X_test)

#Prediction of Train set
ypred = clf.predict(X_train)

#Equalize the shape of Test and Train RDOS values
size = int(ypred.shape[0])
delta = abs(predictions[:size] - ypred[:size])

#print(y_train)

#Calculate Informative Distance
ans = clf.infodist(X_train, delta)

print("Our Implementation predicts: " + str(ans))

print(accuracy(y_test[:size],ans))

#Normal kNN Classification (nCLF)
nCLF = KNeighborsClassifier(n_neighbors = 4)
nCLF.fit(X_train, y_train)
npred = nCLF.predict(X_test)

print("Normal kNN prediction: " + str(npred))#str(Counter(npred).most_common(1)))


print(accuracy(y_test[:size],npred[:size]))

