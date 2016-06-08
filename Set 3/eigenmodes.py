import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from random import random
from math import erfc, erf, sqrt, log10, cos
from tqdm import trange
from scipy import sparse
from scipy import linalg
import scipy.sparse.linalg as spla
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Function to turn the standard square domain into a circular domain
def circDomain(size, M):
    for i in range(size):
        for j in range(size):
            if sqrt((i-size/2.)**2 + (j-size/2.)**2) > size/2:
                M[(i*size + j)] = 0
                M[:,(i*size + j)] = 0
                # M = np.delete(M,(i*size + j),0)
    return M

def rectDomain(size, M, frac):
    for i in range(size):
        for j in range(size):
            if i > size*frac:
                M[(i*size + j)] = 0
                M[:,(i*size + j)] = 0
                # M = np.delete(M,(i*size + j),0)
    return M

def solveSystem(matSize, L, switchS, numEig, shape="r"):
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
        L = 2 * L    #instead of a domain of Lx.5L we want 2LxL
    if switchS:
        M = sparse.csc_matrix(M)
        w, v = spla.eigs(M, numEig, sigma=0)
    else:
        w, v = linalg.eig(M)

    dx= L/matSize
    w = -w/(dx*dx)
    return (w, v.transpose().real)

def sortEigenmodes(w,v):
    wOrder = np.argsort(w)
    w,v = np.sort(w), v[wOrder]
    wOrder = np.nonzero(w)
    w,v = w[wOrder], v[wOrder]
    w,v = w[::1],v[::1]
    return w,v

def plotEigenmode(w,v,eig):
    eigmod = v[eig].reshape((sqrt(len(v[eig])),sqrt(len(v[eig]))))

    plt.title("mode %d \n f=%.4f" %(eig + 1, w[eig].real))
    plt.imshow(eigmod.real, cmap="hot")

def plot_shape(latSize, L, nlist, shape):
    w,v = solveSystem(latSize, L, False, latSize*2, shape)
    w,v = sortEigenmodes(w,v)
    fig = plt.figure()
    fig.set_size_inches(8, 8)
    ax = plt.axes(frameon=False)
    for i in range(len(nlist)):
        plt.subplot(len(nlist)/3,3,i+1)
        plotEigenmode(w,v, nlist[i])
    plt.tight_layout()
    plt.savefig("../../eigenmodes_%s.png" %shape, dpi=300)

def plot_eigenmodes():
    nlist = [0,1,2,3,6,9,19,29,39]
    latSize = 30
    L = 1
    plot_shape(latSize, L, nlist, "s")
    plot_shape(latSize, L, nlist, "c")
    plot_shape(latSize, L, nlist, "r")

def plot_L_dependence():
    latSize = 20
    LList = list(np.arange(1,3.1,0.1))
    shapeList = ["s", "r", "c"]
    eigList = np.zeros((len(shapeList),len(LList)))
    for s in trange(len(shapeList)):
        for l in trange(len(LList)):
            w,v = solveSystem(latSize, LList[l], False, latSize*2, shapeList[s])
            w,v = sortEigenmodes(w,v)
            eigList[s,l] = w[0].real

    plt.plot(LList, eigList[0],"r--", label="square (LxL)")
    plt.plot(LList, eigList[1],"b--",label="rectangle(2LxL)")
    plt.plot(LList,eigList[2],"g--",label="circle(r=.5L)")
    plt.plot(LList, [25/(i*i) for i in LList],"k-", label=r"$25/L^2$")
    plt.title("Frequency dependence on domain size")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.xlim(xmax=3)
    plt.legend()
    plt.savefig("../../L_dependence.png", dpi=300)

def plot_dstep_dependence():
    L = 1
    dList = list(range(5,71,5))
    shapeList = ["s", "r", "c"]
    eigList = np.zeros((len(shapeList),len(dList)))
    for s in trange(len(shapeList)):
        for d in trange(len(dList)):
            w,v = solveSystem(dList[d], L, False, dList[d]*2, shapeList[s])
            w,v = sortEigenmodes(w,v)
            eigList[s,d] = w[0].real
    plt.plot(dList, eigList[0],"r--", label="square (LxL)")
    plt.plot(dList, eigList[1],"b--",label="rectangle(2LxL)")
    plt.plot(dList,eigList[2],"g--", label="circle(r=.5L)")
    plt.title("Frequency dependence on discretization step")
    plt.xlabel("Discretization steps")
    plt.ylabel("Frequency")
    plt.ylim(ymin=0)
    plt.legend(loc=4)
    plt.savefig("../../dstep_dependence.png", dpi=300)


def plot_time():
    latSize = 30
    eig = 7

    w,v = solveSystem(latSize, 1, False, latSize*2, "s")
    w,v = sortEigenmodes(w,v)
    zmin = min(v[eig].real)
    zmax = max(v[eig].real)
    eigmod = v[eig].reshape((sqrt(len(v[eig])),sqrt(len(v[eig]))))
    eigfreq = w[eig].real
    wlength = 1/eigfreq
    print(eigfreq,wlength/9)
    times = list(np.arange(0,wlength*4,wlength*4/9))

    fig = plt.figure()
    ax = plt.axes(frameon=False)
    for t in range(len(times)):
        plt.subplot(len(times)/3, 3, t+1)
        time_step_eigmod = eigmod * cos(eigfreq * times[t])
        print(eigmod)
        plt.imshow(time_step_eigmod.real, cmap="hot", vmin=zmin, vmax=zmax)
        plt.title(times[t])
        ax.xaxis.set_visible(False)
        # plt.tight_layout()
    # plt.savefig("../../eigenmodes_%s.png" %shape, dpi=300)
    plt.show()


# def plot_times():
#     w,v = solveSystem(latSize, 1, False, latSize*2, shapes[0])
#     w,v = sortEigenmodes(w,v)
#     eigmod = v[1].reshape((sqrt(len(v[1])),sqrt(len(v[1]))))
#     eigmod = eigmod

#     fig = plt.figure()
#     # fig.set_size_inches(8, 8)
#     ax = plt.axes(frameon=False)
#     for t in range(len(times)):
#         for s in range(len(shapes)):
#             plt.subplot(len(times)/3,3,t+1)
#             plot_time_step([eigmod],w[1].real, times[t])
#             plt.tight_layout()
#     # plt.savefig("../../eigenmodes_%s.png" %shape, dpi=300)
#     ax.title(w[1].real)
#     plt.show()


## Functions for potential animation
# def init():
#     u_im.set_array(np.ma.array(u))
#     return u_im

# def animate():
#     global u
#     u = step(u)
#     u_im.set_array(u)
#     # print(frame_number)
#     return u_im

# def start_animation():
#     ani = animation.funcanimation(fig, animate, none, init_func=init, interval=10, blit=true)
#     plt.show()

if __name__=="__main__":

    plot_eigenmodes()

    # plot_L_dependence()

    # plot_dstep_dependence()

    # plot_time()
