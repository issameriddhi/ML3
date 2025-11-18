import numpy as np

# Simple matrix
A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
], dtype=float)

# SVD
U, S, VT = np.linalg.svd(A)

print("Matrix A:\n", A)
print("\nU (left singular matrix):\n", U)
print("\nSingular values S:\n", S)
print("\nV^T (right singular matrix):\n", VT)

# Convert S (1D array) into diagonal matrix for reconstruction
S_diag = np.zeros((U.shape[1], VT.shape[0]))
np.fill_diagonal(S_diag, S)

# Reconstruct A
#@ = matrix multiplication
#diag(S) = diagonal matrix made from singular values
A_recon = U @ S_diag @ VT  #
print("\nReconstructed A:\n", A_recon)
