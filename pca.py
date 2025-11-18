import pandas as pd

data ={
    'X1': [2, 4, 5, 6, 8],
    'X2': [3, 5, 7, 9, 11],
    'X3': [1, 1, 2, 3, 5]
}

df =pd.DataFrame(data)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaleddata = scaler.fit_transform(df)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pcadata = pca.fit_transform(scaleddata)

print("Orignal data ",df)
print("pca component",pca.components_)
print("\nExplained Variance Ratio:\n", pca.explained_variance_ratio_)
print("\nPCA Transformed Data:\n", pcadata)

import matplotlib.pyplot as plt

plt.scatter(pcadata[:,0],pcadata[:,1],color="red")
plt.show()  