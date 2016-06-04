# -*- coding: utf-8 -*-
from math import sqrt, sin, pi

import numpy as np
import matplotlib.pyplot as plt


N = 200
assert N % 2 == 0, "N should be even"
# potential is given for -a <= x <= a
a = 1
dx = 2 * a / N
V0 = 10
hbar = m = 1


def x(i):
    return -a + i * dx


def potential(i, p):
    if p == "infinite":
        return 0
    elif p == "finite":
        return V0 if i in (0, N) else 0
    elif p == "finite_scaled":
        return 0 if N // 10 <= i <= 9 * N // 10 else V0
    elif p == "parabolic":
        b = 500
        return b * x(i) ** 2


def solve(p):
    """Solve the time-independent Schrödinger equation with potential p.
    Returns the energies and wave functions.
    """
    A = np.zeros((N, N))

    A[0, 0] = 2 + 2 * dx ** 2 * potential(0, p)
    A[0, 1] = -1

    A[-1, -1] = 2 + 2 * dx ** 2 * potential(N, p)
    A[-1, -2] = -1

    for i in range(1, N - 1):
        A[i, i - 1] = A[i, i + 1] = -1
        A[i, i] = 2 + 2 * dx ** 2 * potential(i, p)

    w, v = np.linalg.eig(A)
    eigenvectors = [v[:, i] for i in range(len(v))]
    w, eigenvectors = zip(*sorted(zip(w, eigenvectors), key=lambda x: x[0]))  # sort by eigenvalues
    energies = [e / (2 * dx ** 2) for e in w]
    return energies, eigenvectors


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


def plot(n, p, psi):
    """Plot probability density for quantum number n."""
    # plt.plot(psi_inf(n) ** 2, label="analytic")
    c1 = "black"
    fig, ax1 = plt.subplots()
    ax1.plot(psi[n - 1] ** 2, label=r"$n$ = %d" % n, color=c1)
    ax1.set_xlabel(r"$i$")
    ax1.set_ylabel(r"$|\psi(x)|^2$", color=c1)
    for t in ax1.get_yticklabels():
        t.set_color(c1)

    ax2 = ax1.twinx()
    c2 = "#5b07ed"
    pot = np.array([potential(i, p) for i in range(N)])
    ax2.plot(pot, label="potential", color=c2, linewidth=4)
    ax2.set_ylabel("potential", color=c2)
    for t in ax2.get_yticklabels():
        t.set_color(c2)

    ncols = 1 if n > 2 else 2
    # ask matplotlib for the plotted objects and their labels, from http://stackoverflow.com/a/10129461
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper center", ncol=ncols)

    ylim = {1: 0.037, 2: 0.027}
    if n in ylim:
        ax1.set_ylim([0, ylim[n]])

    plt.title(r"Time-independent Schrödinger: $n = %d$" % n)
    plt.show()
    # plt.savefig("%s_%d" % (p, n))
    plt.close()


def plot_energies(n=N):
    analytic = [energy(n) for n in range(1, n + 1)]
    infinite = solve("infinite")[0][:n]
    # finite = solve("finite_scaled")[0][:n]
    parabolic = solve("parabolic")[0][:n]

    quantum = range(1, n + 1)
    s = 1

    fig = plt.figure(figsize=(8, 5))
    # plt.scatter(quantum, finite, color="#5d8715", s=s, label="finite")
    plt.scatter(quantum, parabolic, color="#155787", s=s, label="parabolic")
    plt.scatter(quantum, analytic, color='b', s=s, label="analytic infinite")
    plt.scatter(quantum, infinite, color='r', s=s, label="infinite")
    plt.grid(c="grey")
    plt.title("Energies for particle in an infinite potential well")
    plt.xlabel(r"$n$")
    plt.ylabel("Energy")
    plt.xlim([0, n])
    plt.ylim([0, 60000])
    plt.legend(loc="upper left")
    plt.show()
    # plt.savefig("energies", dpi=400)
