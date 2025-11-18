from sklearn.neighbors import KNeighborsClassifier

X = [[1,2], [2,3], [3,3], [8,8], [9,9]]
y = [0, 0, 0, 1, 1]

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

sample = [[2,2]]
print("Prediction:", knn.predict(sample)[0])




