import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from math import erfc, erf, sqrt, log10
from random import random

from tqdm import trange

latSize = 256 + 2               # lattice + padding
growthPar = 1.5
omega = 1
steps = 2000
nutLattice = np.zeros((latSize,latSize))
nutLattice[0] = 1.              # top boundary
objLattice =np.zeros((latSize,latSize))
growCand = []

def sor_object(c1, obj, tolerance, omega):
    c0 = np.copy(c1)
    deltas = []
    delta = float("Infinity")
    while (delta >= tolerance):
        for i in range(1, latSize-2):
            for j in range(latSize-1):
                jp = j + 1 if j < latSize else 0    # periodic            conditions
                jm = j - 1 if j > 0 else latSize - 1    #           boundary
                if obj[i, j] == 1:
                    c1[i, j] = 0  # object
                else:
                    c1[i, j] = ((1 - omega) * c0[i, j] +
                                omega / 4 * (c0[i + 1, j] + c1[i - 1, j] + c0[i, jp] + c1[i, jm]))

        delta = np.max(np.abs(c1 - c0))
        deltas.append(delta)
        # print(delta)
        c0[1:latSize] = c1[1:latSize]
    return c1, deltas

def grow_step(objLattice, nutLattice):
    objLatticeOld = np.copy(objLattice)
    growthNorm = sum(nutLattice[i, j] ** growthPar
                     for i in range(1, latSize - 1) for j in range(1, latSize - 1)
                     if check_connection(objLattice,i,j))
    for i in range(1,latSize-1):
        for j in range(1,latSize-1):
            if not objLattice[i,j]:
                if check_connection(objLatticeOld, i, j):
                    if nutLattice[i,j] < 0:
                        print("negative nutrient!", nutLattice[i,j])
                    prob = (nutLattice[i,j] ** growthPar) / growthNorm
                    if random() < prob:
                        objLattice [i,j] = 1
    return objLattice

def check_connection(lattice, i, j):
    nbrTot = lattice [i-1,j] + lattice [i+1,j] + lattice [i,j-1] + lattice[i,j+1]
    return (nbrTot > 0)

def grow(objLattice, nutLattice, tolerance, omega, steps):
    objLattice[latSize - 10, (latSize- 1)//2] = 1
    for n in trange(steps):
        # print(n, end=" ", flush=True)
        nutLattice, deltas = sor_object(nutLattice, objLattice, tolerance, omega)
        objLattice = grow_step(objLattice, nutLattice)

    return objLattice, nutLattice

def initNutLattice(nutLattice):
    for i in range(latSize):
        nutLattice[i] = 1 - (i/latSize)
    nutLattice[latSize - 10, (latSize- 1)//2] = 0
    return nutLattice

nutLattice = initNutLattice(nutLattice)
obj, nut = grow(objLattice, nutLattice, 1E-1, omega, steps)
plt.imshow(nut + obj)
plt.show()
