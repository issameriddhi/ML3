import random
import numpy as np

x = np.array([2,2,5,4,8,6,5,5,6,4])
y = np.array([4,6,6,7,3,6,2,7,3,4])

data = np.column_stack((x,y)) 

# Random initial centroids
m1 = random.choice(data)
m2 = random.choice(data)

centroids = np.array([m1, m2], dtype=float)

it = 1

while True:
    print("\n==============================")
    print(f"        Iteration {it}")
    print("==============================")
    print(f"Old Centroid 1: {centroids[0]}")
    print(f"Old Centroid 2: {centroids[1]}\n")

    k1 = []
    k2 = []

    print("Point        d(C1)      d(C2)      Assigned")
    print("--------------------------------------------")

    # Step 1: assign points
    for p in data:
        d1 = np.linalg.norm(p - centroids[0])
        d2 = np.linalg.norm(p - centroids[1])

        if d1 < d2:
            k1.append(p)
            assigned = "C1"
        else:
            k2.append(p)
            assigned = "C2"

        print(f"{p}   {d1:8.3f}   {d2:8.3f}      {assigned}")

    print("\nCluster 1:", k1)
    print("Cluster 2:", k2)

    # Step 2: new centroids AFTER loop
    c1 = np.mean(k1, axis=0) if len(k1) > 0 else centroids[0]
    c2 = np.mean(k2, axis=0) if len(k2) > 0 else centroids[1]

    print("\nNew Centroid 1:", c1)
    print("New Centroid 2:", c2)

    # Step 3: stabilization check
    if np.allclose(centroids[0], c1) and np.allclose(centroids[1], c2):
        print("\n✔ Centroids stabilized. K-Means complete!")
        break

    # Step 4: update centroids
    centroids[0] = c1
    centroids[1] = c2
    it += 1
