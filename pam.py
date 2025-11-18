import numpy as np, math, random, matplotlib.pyplot as plt

dataset = np.array([
    [2,4],[8,2],[9,3],[1,5],[8.5,1],
    [6,5],[8,4],[6,8],[1,2],[3,6],
    [7,5],[8.5,1],[9,3]
])
K = 2
random.seed(0)

def dist(a,b): return math.dist(a,b)
def cost(m, data): return sum(min(dist(p,mm) for mm in m) for p in data)
def assign(m, data): return np.array([np.argmin([dist(p,mm) for mm in m]) for p in data])

idx = random.sample(range(len(dataset)), K)
medoid_idx = idx.copy()
medoids = dataset[medoid_idx]
curr_cost = cost(medoids, dataset)

improve = True
while improve:
    improve = False
    for i,m in enumerate(medoid_idx):
        for c in range(len(dataset)):
            if c in medoid_idx: continue
            temp = medoid_idx.copy(); temp[i] = c
            newm = dataset[temp]
            newc = cost(newm, dataset)
            if newc < curr_cost:
                medoid_idx, medoids, curr_cost = temp, newm, newc
                improve = True
                break
        if improve: break

labels = assign(medoids, dataset)

plt.scatter(dataset[:,0], dataset[:,1], c=labels, cmap='viridis', s=60)
plt.scatter(medoids[:,0], medoids[:,1], c='red', marker='X', s=200)
plt.title("PAM Clustering")
plt.grid(True)
plt.show()

print("Medoids:\n", medoids)
print("Final Cost:", round(curr_cost,3))
print("Cluster sizes:", [sum(labels==i) for i in range(K)])
