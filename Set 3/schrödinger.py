# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, sin, pi


N = 200
assert N % 2 == 0, "N should be even"
# potential is given for -a <= x <= a
a = 1
dx = 2 * a / N
V0 = 1e6
hbar = m = 1
p = "infinite"


def x(i):
    return -a + i * dx

A = np.zeros((N, N))

def potential(i):
    if p == "infinite":
        return 0
    elif p == "finite":
        return V0 if i in (0, N) else 0
    elif p == "parabolic":
        b = dx
        return b * x(i) ** 2


A[0, 0] = 2 + 2 * dx ** 2 * potential(0)
A[0, 1] = -1

A[-1, -1] = 2 + 2 * dx ** 2 * potential(N)
A[-1, -2] = -1

for i in range(1, N - 1):
    A[i, i - 1] = A[i, i + 1] = -1
    A[i, i] = 2 + 2 * dx ** 2 * potential(i)


w, v = np.linalg.eig(A)
eigenvectors = [v[:, i] for i in range(len(v))]
w, v = zip(*sorted(zip(w, eigenvectors), key=lambda x: x[0]))  # sort by eigenvalues
energies = [e / (2 * dx ** 2) for e in w]


def psi_inf(n):
    """Analytic solution infinite potential well. Adapted from eq. 2.28,
    'Introduction to Quantum Mechanics', D. Griffiths.
    """
    v = np.array([sqrt(2 / a) * sin(n * pi * (i - a) / (2 * a)) for i in np.arange(-a, a + dx, dx)])
    v = v / np.linalg.norm(v)   # normalize
    return v


def energy(n):
    """Analytic energies for infinite potential well. Adapted from Eq. 2.27 from
    Griffiths.
    """
    return (n * pi * hbar / (2 * a)) ** 2 / (2 * m)


def plot(n):
    """Plot probability density for quantum number n."""
    plt.plot(psi_inf(n) ** 2, label="analytic")
    plt.plot(v[n - 1] ** 2, label="discrete")
    pot = np.array([potential(i) for i in range(N)])
    pot = pot / np.linalg.norm(pot)
    plt.plot(pot, label="potential")
    plt.legend(ncol=3)
    plt.title(r"Time-independent Schrödinger: $n = %d$" % n)
    plt.xlabel(r"$i$")
    plt.ylabel(r"$|\psi(x)|^2$")
    # plt.show()
    plt.savefig("%s_%d" % (p, n))
    plt.close()


def plot_energies(n=N):
    real = [energy(n) for n in range(1, n + 1)]
    calculated = energies[:n]
    assert len(real) == len(calculated)
    diff = [real[i] - calculated[i] for i in range(n)]
    quantum = range(1, n + 1)
    s = 1

    fig = plt.figure(figsize=(8, 5))
    plt.scatter(quantum, real, color='b', s=s, label="analytic")
    plt.scatter(quantum, calculated, color='r', s=s, label="calculated")
    plt.grid(c="grey")
    plt.title("Energies for particle in the infinite potential well")
    plt.xlabel(r"$n$")
    plt.ylabel("Energy")
    plt.xlim([0, n])
    plt.ylim([0, 60000])
    plt.legend(loc="upper left")
    # plt.show()
    plt.savefig("energies", dpi=400)
