import numpy as np
import math
import matplotlib.pyplot as plt
import random

dataset = np.array([
    [2, 4], [8, 2], [9, 3], [1, 5], [8.5, 1], [6, 5], [8, 4], [6, 8], [1, 2], [3, 6], [7, 5], [8.5, 1], [9, 3]
])
K = 2
SAMPLE_SIZE = 4

def dist(p1, p2):
    return math.dist(p1, p2)

data_indices = list(range(len(dataset)))
sample_indices = random.sample(data_indices, SAMPLE_SIZE)

# Simplified Medoid Selection (Logic based on first K indices)
initial_medoid_indices = sample_indices[:K] 
medoids = dataset[initial_medoid_indices]

assignments = []
total_cost = 0

for point in dataset:
    distances = [dist(point, medoid) for medoid in medoids]
    closest_medoid_index = np.argmin(distances)
    assignments.append(closest_medoid_index)
    total_cost += min(distances) # Cost is the distance to the medoid

assignments = np.array(assignments)
final_cost = round(total_cost, 3)

plt.figure(figsize=(7, 6))

# 1. Plot the clustered points, colored by assignment
scatter = plt.scatter(dataset[:, 0], dataset[:, 1], c=assignments, cmap='viridis', s=50, alpha=0.7) 

# 2. Plot Medoids (The 'X' markers)
plt.scatter(medoids[:, 0], medoids[:, 1], c='red', marker='X', s=200, label='Medoids') 

# 3. Create Custom Legend for Cluster Names
# We use the unique assignments (0, 1) to create color handles
legend_elements = [plt.scatter([], [], marker='o', color=scatter.cmap(scatter.norm(c)), label=f'Cluster {c+1}') 
                   for c in np.unique(assignments)]
legend_elements.append(plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markersize=10, label='Medoid'))

plt.title(f"CLARA Result (K={K})")
plt.legend(handles=legend_elements)
plt.grid(True)
plt.show()

print(f"Final Cost: {final_cost}")