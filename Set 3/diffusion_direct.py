# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy import linalg
import scipy.sparse.linalg as spla

from eigenmodes import circDomain


L = 4                           # from -2 to 2
N = 80
dx = L / N
Ar = np.zeros((1, N))
Ar[0, 0] = -4
Ar[0, 1] = 1
A = sparse.block_diag([linalg.toeplitz(Ar) for m in range(N)])
B = sparse.diags([1, 1], [N, -N], (N ** 2, N ** 2))
M = A + B
M = M.toarray()
M = -M

radius = 2

def circle(M):
    for i in range(N):
        for j in range(i, N):
            x = -L / 2 + i * dx
            y = -L / 2 + j * dx
            if np.sqrt(x ** 2 + y ** 2) > radius:
                M[i * N + j, :] = 0
                M[:, i * N + j] = 0
    return M

def circle_domain(B):
    for i in range(N):
        for j in range(i, N):
            x = -L / 2 + i * dx
            y = -L / 2 + j * dx
            if np.sqrt(x ** 2 + y ** 2) > radius:
                B[i, j] = 0
                B[j, i] = 0
    return B

b = np.zeros(N ** 2)
source = (N / 2 + 0.6 / dx) * N + (N / 2 + 1.2 / dx)
b[round(source)] = 1

# M = circle(M)
x = np.linalg.solve(M, b)
x_im = x.reshape(N, N)

fig, ax = plt.subplots()
plt.imshow(x_im, cmap="hot")
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
plt.colorbar()
plt.title("Solution to the Diffusion Equation")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.savefig("diffusion", dpi=400)
