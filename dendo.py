import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
import pandas as pd

data = pd.DataFrame({
    'X': [1, 2, 3, 8, 9],
    'Y': [1, 1, 2, 8, 9]
})

#data = np.array([1, 2, 3, 8, 9]).reshape(-1, 1) 1-d

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.title("Single Linkage")
shc.dendrogram(shc.linkage(data, method='single'))

plt.subplot(2, 2, 2)
plt.title("Complete Linkage")
shc.dendrogram(shc.linkage(data, method='complete'))

plt.subplot(2, 2, 3)
plt.title("Average Linkage")
shc.dendrogram(shc.linkage(data, method='average'))

plt.subplot(2, 2, 4)
plt.title("Ward Linkage")
shc.dendrogram(shc.linkage(data, method='ward'))

plt.tight_layout()
plt.show()
