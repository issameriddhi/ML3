import numpy as np

# ----------------------------------------------------------
#  INPUT DATA (as given in your book)
# ----------------------------------------------------------
class1 = np.array([[2,3],[3,3],[4,5]])    # x1, x2, x3
class2 = np.array([[6,8],[7,9],[8,7]])    # p1, p2, p3

X = np.vstack([class1, class2])
y = np.array([0,0,0, 1,1,1])  # 0 = class1, 1 = class2

print("=== INPUT DATA ===")
print("Class 1:\n", class1)
print("Class 2:\n", class2)

# ----------------------------------------------------------
# 1. Compute CLASS MEANS
# ----------------------------------------------------------
m0 = class1.mean(axis=0)
m1 = class2.mean(axis=0)

print("\n=== CLASS MEANS ===")
print("Mean of Class 1 (m0):", m0)
print("Mean of Class 2 (m1):", m1)

# ----------------------------------------------------------
# 2. Compute WITHIN-CLASS SCATTER MATRICES S0, S1
# ----------------------------------------------------------
S0 = np.zeros((2,2))
for x in class1:
    S0 += np.outer(x - m0, x - m0)

S1 = np.zeros((2,2))
for x in class2:
    S1 += np.outer(x - m1, x - m1)

Sw = S0 + S1

print("\n=== SCATTER MATRICES ===")
print("S0 (Class 1):\n", S0)
print("S1 (Class 2):\n", S1)
print("Sw = S0 + S1:\n", Sw)

# ----------------------------------------------------------
# 3. Compute LDA DIRECTION w = Sw^-1 (m1 - m0)
# ----------------------------------------------------------
w = np.linalg.inv(Sw).dot(m1 - m0)

print("\n=== LDA VECTOR ===")
print("w (unnormalized):", w)

# ----------------------------------------------------------
# 4. Project all points onto w (makes data 1D)
# ----------------------------------------------------------
X_proj = X.dot(w)

print("\n=== 1D PROJECTIONS ===")
for i, (pt, proj) in enumerate(zip(X, X_proj)):
    cls = "Class 1" if y[i] == 0 else "Class 2"
    print(f"{pt} → {proj:.4f}  ({cls})")

# ----------------------------------------------------------
# 5. Compute DECISION THRESHOLD (midpoint)
# ----------------------------------------------------------
m0_proj = m0.dot(w)
m1_proj = m1.dot(w)
threshold = (m0_proj + m1_proj) / 2

print("\nMean projection of Class 1:", m0_proj)
print("Mean projection of Class 2:", m1_proj)
print("Decision threshold:", threshold)

# ----------------------------------------------------------
# 6. CLASSIFY TEST POINT (5,6)
# ----------------------------------------------------------
test = np.array([5,6])
test_proj = test.dot(w)

print("\n=== CLASSIFICATION OF TEST POINT (5,6) ===")
print("Projection of test point:", test_proj)

label = 1 if test_proj > threshold else 0
print("Predicted Class:", label, "(1 = Class 2, 0 = Class 1)")
