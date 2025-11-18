import math

X = [[1, 2], [2, 3], [3, 3], [8, 8], [9, 9]]
y = [0, 0, 0, 1, 1]

def knn_predict(X, y, sample, k=3):
    distances = []

    for i in range(len(X)):
        d = math.dist(sample, X[i])
        distances.append((d, y[i]))


    distances.sort(key=lambda x: x[0])


    k_labels = [label for (dist, label) in distances[:k]]

    prediction = max(set(k_labels), key=k_labels.count)
    return prediction

sample = [2, 2]
result = knn_predict(X, y, sample)

print("Prediction:", result)
