import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from random import random
from math import erfc, erf, sqrt, log10
from tqdm import trange
from scipy import sparse
from scipy import linalg
import scipy.sparse.linalg as spla

latSize = 40
eigNum = 3

# Initialize finite difference matrix blocks
Ar = np.zeros((1, latSize))
Ar[0,0] = -4
Ar[0,1] = 1

# Function to turn the standard square domain into a circular domain
def circDomain (latSize, M):
    for i in range(latSize):
        for j in range(latSize):
            if sqrt((i-latSize/2.)**2 + (j-latSize/2.)**2) > latSize/2:
                M[(i*latSize + j)] = 0
    return M

def rectDomain (latSize, M, frac):
    for i in range(latSize):
        for j in range(latSize):
            if sqrt((i-latSize/2.)**2 + (j-latSize/2.)**2) > latSize/2:
                M[(i*latSize + j)] = 0
    return M

def solveSystem (latSize, switchS, shape="s"):
    """
    Solve the eigenvalue problem for system of a certain shape and
    latSize using either dense or sparse matrix methods. Shape is
    given as a single character "s" = square, "c" = circle, "r" =
    rectangle.
    Returns eigenvalues w array(1,M) and eigenvectors of length
    latSize**2
    """

    A = sparse.block_diag([linalg.toeplitz(Ar) for m in range(latSize)])
    B = sparse.diags([1,1],[latSize,-latSize], (latSize**2, latSize**2))
    M = A + B
    M = M.toarray()
    print(M)
    if shape is "c":
        M = circDomain(latSize, M)
    elif shape is "r":
        M = rectDomain(latSize, M, 0.5)
    if switchS:
        M = sparse.csr_matrix(M)
        (w, v) = spla.eigs(M, latSize)
    else:
        (w, v) = linalg.eig(M)
    return (w, v)

(w,v) = solveSystem(latSize, False)
eigmod = v[:,eigNum].reshape((latSize,latSize))
print(w)
plt.title("Eigenmode #%d \n Frequency: %d" %(eigNum, w[eigNum].real) )
plt.imshow(eigmod.real, cmap="hot")
plt.colorbar()
plt.show()
