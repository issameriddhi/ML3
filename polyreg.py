import pandas as pd
import numpy as np
from sklearn.linear_model import  LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


data = pd.read_csv("log.csv")

x = np.array(data["x"]).reshape(-1,1)
y = data["y"]

poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x)

model = LinearRegression()

model.fit(x_poly,y)

j = np.array([2]).reshape(-1,1)
u = poly.fit_transform(j)
print("for 2 ",model.predict(u))

xv = np.linspace(min(x),max(x),100).reshape(-1,1)
xval = poly.transform(xv) 
yp = model.predict(xval)

plt.scatter(x,y,color="red")
plt.plot(xv,yp,color="blue")
plt.show()