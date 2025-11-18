import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("log.csv")

x = np.array(data["x"]).reshape(-1,1)
y = data["y"]

# Add bias
X = np.hstack((np.ones_like(x), x))

# Initialize weights
weights = np.zeros(X.shape[1])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Training
lr = 0.1
epoch = 1000

for _ in range(epoch):
    z = X.dot(weights)
    pred = sigmoid(z)
    error = y - pred
    fixer = X.T.dot(error) / len(y)
    weights += lr * fixer

# Predict smooth curve
xl = np.linspace(0, 10, 100).reshape(-1,1)
XL = np.hstack((np.ones_like(xl), xl))
z_pred = XL.dot(weights)
yp = sigmoid(z_pred)

# Plot
plt.scatter(x, y, color="blue", label="Actual Data")
plt.plot(xl, yp, color="red", label="Logistic Curve")
plt.legend()
plt.show()
