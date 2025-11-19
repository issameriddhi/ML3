import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# ------- Create a simple linear dataset -------
X = np.array([
    [1,2],[2,3],[3,4],[4,5],   # Class 0
    [5,1],[6,2],[7,3],[8,4]    # Class 1
])
y = np.array([0,0,0,0,1,1,1,1])

# ----------- Train Linear SVM -----------
linear_model = SVC(kernel='linear')
linear_model.fit(X, y)

# ----------- Train RBF SVM --------------
rbf_model = SVC(kernel='rbf')
rbf_model.fit(X, y)

# --------- Create mesh/grid for plotting ---------
xx, yy = np.meshgrid(np.linspace(0,9,200), np.linspace(0,9,200))
grid = np.c_[xx.ravel(), yy.ravel()]

# Predictions
linear_pred = linear_model.predict(grid).reshape(xx.shape)
rbf_pred = rbf_model.predict(grid).reshape(xx.shape)

# ---------------- Plot Linear Kernel ----------------
plt.figure(figsize=(6,5))
plt.contourf(xx, yy, linear_pred, alpha=0.3)
plt.scatter(X[:,0], X[:,1], c=y, edgecolors='black')
plt.title("SVM with Linear Kernel")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# ---------------- Plot RBF Kernel -------------------
plt.figure(figsize=(6,5))
plt.contourf(xx, yy, rbf_pred, alpha=0.3)
plt.scatter(X[:,0], X[:,1], c=y, edgecolors='black')
plt.title("SVM with RBF Kernel")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
