import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np 
import matplotlib.pyplot as plt

data = pd.read_csv("log.csv")

x = np.array(data['x']).reshape(-1,1)
y = data['y']
print(x)
model = LogisticRegression()

model.fit(x,y)

xval = np.linspace(0,10,100).reshape(-1,1)
prediction = model.predict_proba(xval)

plt.scatter(x,y,color="blue",label="actucal data")
plt.plot(xval,prediction[:,1],color="red")
plt.legend()
plt.show()
