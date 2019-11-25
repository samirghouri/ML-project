import math
import numpy as np
from statistics import median
from collections import Counter


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

# def predict(self, X):
#     predictions = [self._predict(x)for x in X]
#     return np.array(predictions)

def get_RDOS(p,x, y_train, k ):
    distance = [euclidean_distance(p, i)for i in x]
    
    k_indices = np.argsort(distance)[:k]
    # print(k_indices)
    k_nearest_labels = [y_train[i]for i in k_indices]

    # most_common = Counter(k_nearest_labels).most_common(1)
    # it returns the most commmon item in form of tuple
    # return most_common[0][0]

    # finding the forward density
    sorted_distance = sorted(distance)
    sorted_distance = list(filter(lambda a: a != 0.00, sorted_distance))
    fd_list = []
    for i in range(k):
        forward_density = (i+1)/sorted_distance[i]
        fd_list.append(forward_density)
    # finding the ranking of x by q
    N_x = []
    for k in k_indices:
        N_x.append(x[k])

    reverse_list = []
    for s in N_x:
        count, R = 0, 0
        if s in x:
            distance2 = [euclidean_distance(
                s, x_train)for x_train in x]
        distance2 = sorted(distance2)
        distance2 = list(filter(lambda a: a != 0.00, distance2))
        for dis in distance2:
            if dis < sorted_distance[count]:
                R += 1
        reverse_list.append(R)
        count += 1
    # calculating the RDOS values
    count = 0
    RDOS = []
    for i in range(4):
        count += 1
        RDOS.append((reverse_list[i]-count)/fd_list[i])
        
    return median(RDOS)


def delta(i,j,x, y_train, k):
    return abs(get_RDOS(i,x, y_train, k)-get_RDOS(j,x, y_train, k))

def infodist(x, y_train, k):
    dist = []
    answer = []
    for i in x:
        k_indices = []
        dist = []
        for j in x:
            dist.append(euclidean_distance(i,j)*(1+math.log(1+delta(i,j,x, y_train, k))))

        k_indices = np.argsort(dist)[:k]
    
        #print(k_indices)
    
        k_nearest_labels = []
    
        for i in k_indices:
            k_nearest_labels.append(y_train[i])
            
        most_common = Counter(k_nearest_labels).most_common(1)
            # it returns the most commmon item in form of tuple
        answer.append(most_common[0][0])

    return answer