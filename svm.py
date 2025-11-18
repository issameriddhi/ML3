import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'x1':[1,2,2,3,3,4,6,7,8,9],
    'x2':[1,1,2,2,3,3,5,6,7,8],
    'y' :[0,0,0,0,0,1,1,1,1,1]
})

X = df[['x1','x2']].values
y = df['y'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

model = SVC(kernel='rbf', probability=True)
model.fit(X_train,y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

x_given = np.array([[4,3]])
print("Prediction for", x_given, "->", model.predict(x_given)[0])
print("Probability:", model.predict_proba(x_given)[0])

# plot
plt.scatter(X[:,0], X[:,1], c=y, s=60, cmap='coolwarm')
plt.scatter(x_given[:,0], x_given[:,1], c='black', marker='X', s=200)
plt.title("SVM Classification")
plt.grid(True)
plt.show()
