import random
import numpy as np

data = [1, 2, 3, 10, 11, 12]

m1 = random.choice(data)
m2 = random.choice(data)

iteration = 1

while True:
    k1 =[]
    k2 =[]
    print(f"{iteration}")

    print(f"centroids {m1} and {m2}")
    for p in data:

        if abs(p-m1) < abs(p-m2):
            k1.append(p)
        else:
            k2.append(p)

    c1 = np.mean(k1) if k1 else m1
    c2 = np.mean(k2) if k2 else m2

    print(f"cluster k1 : {k1} centroid : {c1}")
    print(f"cluster k2 : {k2} centroid : {c2}")

    if c1 == m1 and c2 == m2 :
        print("centroids are satablized")
        break
        
    m1 = c1
    m2 = c2
    iteration +=1