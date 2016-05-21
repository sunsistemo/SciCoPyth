import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from random import random
from math import erfc, erf, sqrt, log10
from tqdm import trange
from scipy import sparse
from scipy import linalg
import numpy.linalg as npla

latSize = 40

# Initialize finite difference matrix blocks
Ar = np.zeros((1, latSize))
Ar[0,0] = 4
Ar[0,1] = -1

# Function to turn the standard square domain into a circular domain
def circDomain (latSize, M):
    for i in range(latSize):
        for j in range(latSize):
            if sqrt((i-latSize/2.)**2 + (j-latSize/2.)**2) > latSize/2:
                M[(i*latSize + j)] = 0
    return M

A = sparse.block_diag([linalg.toeplitz(Ar) for m in range(latSize)])
B = sparse.diags([-1,-1],[latSize,-latSize], (latSize**2, latSize**2))
M = A + B
M = M.toarray()
M = circDomain(latSize, M)
(w, v) = npla.eig(M)
eigmod = v[:,89].reshape((latSize,latSize))

plt.imshow(eigmod.real, cmap="hot")
plt.colorbar()
plt.show()
