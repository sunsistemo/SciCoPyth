import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from random import random
from math import erfc, erf, sqrt, log10
from tqdm import trange
from scipy import sparse
from scipy import linalg
import scipy.sparse.linalg as spla

# Function to turn the standard square domain into a circular domain
def circDomain (size, M):
    for i in range(size):
        for j in range(size):
            if sqrt((i-size/2.)**2 + (j-size/2.)**2) > size/2:
                M[(i*size + j)] = 0
                # M = np.delete(M,(i*size + j),0)
    return M

def rectDomain (size, M, frac):
    for i in range(size):
        for j in range(size):
            if i > size*frac:
                M[(i*size + j)] = 0
                # M = np.delete(M,(i*size + j),0)
    return M

def solveSystem (matSize, switchS, numEig, shape="r"):
    """
    Solve the eigenvalue problem for system of a certain shape and
    matSize using either dense or sparse matrix methods. Shape is
    given as a single character "s" = square, "c" = circle, "r" =
    rectangle.
    Returns eigenvalues w array(1,M) and eigenvectors of length
    matSize**2
    """
    Ar = np.zeros((1, matSize))
    Ar[0,0] = -4
    Ar[0,1] = 1
    A = sparse.block_diag([linalg.toeplitz(Ar) for m in range(matSize)])
    B = sparse.diags([1,1],[matSize,-matSize], (matSize**2, matSize**2))
    M = A + B
    M = M.toarray()
    if shape is "c":
        M = circDomain(matSize, M)
    elif shape is "r":
        M = rectDomain(matSize, M, 0.5)
    if switchS:
        M = sparse.csc_matrix(M)
        w, v = spla.eigs(M, numEig, sigma=0)
    else:
        w, v = linalg.eig(M)
    return (w, v.transpose())

def sortEigenmodes(w,v):
    wOrder = np.argsort(w)
    w,v = np.sort(w), v[wOrder]
    wOrder = np.nonzero(w)
    w,v = w[wOrder], v[wOrder]
    w,v = w[::-1],v[::-1]
    return w,v

def plotEigenmode(w,v,eig):
    eigmod = v[eig].reshape((sqrt(len(v[eig])),sqrt(len(v[eig]))))

    plt.title("Eigenmode #%d \n Frequency: %f" %(eig + 1, -w[eig].real))
    plt.imshow(eigmod.real, cmap="hot")
    # plt.colorbar()
    # plt.show()

def init():
    u_im.set_array(np.ma.array(u))
    return u_im


def animate():
    global u
    u = step(u)
    u_im.set_array(u)
    # print(frame_number)
    return u_im

def start_animation():
    ani = animation.FuncAnimation(fig, animate, None, init_func=init, interval=10, blit=True)
    plt.show()


if __name__=="__main__":
    latSize = 100
    eigNum = 50

    w,v = solveSystem(latSize, False, latSize*2, "c")
    w,v = sortEigenmodes(w,v)
    nlist = [0,1,2,3,10,20]
    plt.figure(1)
    for i in range(len(nlist)):
        plt.subplot(2,len(nlist)/2,i+1)
        plotEigenmode(w,v, nlist[i])
    plt.show()
