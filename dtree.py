import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

# ---------------------------------------
# Dataset
# ---------------------------------------
df = pd.DataFrame({
    "Outlook": ["Sunny","Sunny","Overcast","Rain","Rain","Rain","Overcast","Sunny","Sunny","Rain","Sunny","Overcast","Overcast","Rain"],
    "Temp": ["Hot","Hot","Hot","Mild","Cool","Cool","Cool","Mild","Cool","Mild","Mild","Mild","Hot","Mild"],
    "Humidity": ["High","High","High","High","Normal","Normal","Normal","High","Normal","Normal","Normal","High","Normal","High"],
    "Wind": ["Weak","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Strong"],
    "Play": ["N","N","Y","Y","Y","N","Y","N","Y","Y","Y","Y","Y","N"]
})

sample = pd.DataFrame({
    "Outlook": ["Sunny"],
    "Temp": ["Hot"],
    "Humidity": ["High"],
    "Wind": ["Weak"]
})

df_enc = df.copy()

# Encode training data
se = sample.apply(LabelEncoder().fit_transform) 
df_encoded = df.apply(LabelEncoder().fit_transform) 
x = df_encoded.drop("Play",axis=1) 
y = df_encoded["Play"]


cart = DecisionTreeClassifier(criterion="gini")
cart.fit(x, y)
cart_pred = cart.predict(se)[0]

id3 = DecisionTreeClassifier(criterion="entropy")
id3.fit(x, y)
id3_pred = id3.predict(se)[0]

nb = GaussianNB()
nb.fit(x, y)
nb_pred = nb.predict(se)[0]


print("CART Prediction  :", cart_pred)
print("ID3 Prediction   :", id3_pred)
print("Naive Bayes      :", nb_pred)

# ---------------------------------------
# Plot Decision Tree (ID3)
# ---------------------------------------
plt.figure(figsize=(10,6))
plot_tree(id3,
          feature_names=x.columns,
          class_names=["No","Yes"],
          filled=True)
plt.show()
