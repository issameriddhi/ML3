import numpy as np
from collections import Counter

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

#for one feature
#def euclidean_distance(a, b):
    #return abs(a - b)


# KNN prediction function
def knn_predict(training_data, training_labels, test_point, k):
    distances = []
    
    # Calculate distance from test point to every training point
    for i in range(len(training_data)):
        dist = euclidean_distance(test_point, training_data[i])
        distances.append((dist, training_labels[i]))
        
    # Sort distances (smallest first)
    distances.sort(key=lambda x: x[0])
    
    # Pick k nearest labels
    k_nearest = [label for _, label in distances[:k]]
    
    # Return the most common label
    return Counter(k_nearest).most_common(1)[0][0]


# ------------------------------
# Dataset
training_data = [
    [1, 50],
    [2, 60],
    [3, 65],
    [6, 80],
    [7, 90]
]

training_labels = ['Fail', 'Fail', 'Pass', 'Pass', 'Pass']

# Test point
test_point = [4, 70]

# Choose k
k = 3

# Predict
prediction = knn_predict(training_data, training_labels, test_point, k)
print("Predicted class:", prediction)
