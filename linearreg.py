import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")

x = data["exp"]
y = data["sal"]

n = int(len(x))

m = (n*(np.sum(x*y)) - np.sum(x) * np.sum(y))/(n*(np.sum(x**2)) - (np.sum(x))**2)
b = (np.sum(y) - m *(np.sum(x)))/n


print("x  |   y |  y_pred")
for i in range(len(x)):
    y_pred = b + m*x[i]
    print(f"{x[i]} |{y[i]} |{y_pred:.2f}")

plt.scatter(x,y,color="blue",label = "ACTUAL DATA")
plt.plot(x,b + m*x,color="red" , label="predicted line")
plt.xlabel("salary")
plt.ylabel("exp")
plt.legend()
plt.show()