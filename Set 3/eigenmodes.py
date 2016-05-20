import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from random import random
from math import erfc, erf, sqrt, log10
from tqdm import trange
from scipy import sparse
from scipy import linalg

latSize = 256

# Initialize finite difference matrix blocks
Ar = np.zeros((1, latSize))
Ar[0,0] = 5
Ar[0,1] = -1
A = sparse.block_diag([linalg.toeplitz(Ar) for m in range(latSize)])
B = sparse.diags([-1,-1],[latSize,-latSize], (latSize**2, latSize**2))
M = A + B
# print(C.toarray())

# Initialize circular domain matrix
C = np.zeros((latSize,latSize))
count = 0
for i in range(latSize):
    for j in range(latSize):
        if sqrt((i-latSize/2.)**2 + (j-latSize/2.)**2) < latSize/2 - 1:
            count = count + 1
            C[i,j] = count
        elif sqrt((i-latSize/2.)**2 + (j-latSize/2.)**2) < latSize/2:
            C[i,j] = -1
        else:
            C[i,j] = 0

# print(C)
plt.imshow(C)
plt.show()
