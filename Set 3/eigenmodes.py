import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from random import random
from math import erfc, erf, sqrt, log10
from tqdm import trange
from scipy import sparse
from scipy import linalg

latSize = 3

# Initialize finite difference matrix blocks
Ar = np.zeros((1, latSize))
Ar[0,0] = 4
Ar[0,1] = -1

# Initialize circular domain matrix
def circDomain(latSize, numbering = False):
    C = np.zeros((latSize,latSize))
    count = 0
    print("hello")
    for i in range(latSize):
        for j in range(latSize):
            if numbering and sqrt((i-latSize/2.)**2 + (j-latSize/2.)**2) < latSize/2 - 1:
                count = count + 1
                C[i,j] = count
            elif sqrt((i-latSize/2.)**2 + (j-latSize/2.)**2) < latSize/2:
                C[i,j] = -1
            else:
                C[i,j] = 0
    return C

# C = circDomain(latSize)
A = sparse.block_diag([linalg.toeplitz(Ar) for m in range(latSize)])
B = sparse.diags([-1,-1],[latSize,-latSize], (latSize**2, latSize**2))
M = A + B
# print(C.toarray())

M = M.todense()
print(M)
I = np.eye(9)
wi, vi = linalg.eig(I)
w, v = linalg.eig(M)
print(I,wi,vi)
# print(v[5].reshape((latSize,latSize), order="C"))
# print(w, v)
print("w*v",np.dot(w[1],v[1]))
print("M*v",np.dot(M,v[1]))
eigvec = v[1].reshape((latSize,latSize))
# print(eigvec)
plt.imshow(eigvec)
plt.colorbar()
plt.show()
