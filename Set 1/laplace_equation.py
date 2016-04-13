# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

N = 50
dx = dy = 1 / N
xs = [i * dx for i in range(N + 1)]
c0 = np.zeros((N + 1, N + 1))
c0[0] = 1.                       # boundary condition at the top
c1 = np.copy(c0)

delta = float("Infinity")
tolerance = 1E-5
steps = 0

def jacobi(c0, c1, delta):
    deltas = []
    while (delta >= tolerance):
        for i in range(1, N):
            for j in range(N + 1):
                jp = j + 1 if j < N else 0    # periodic            conditions
                jm = j - 1 if j > 0 else N    #           boundary
                c1[i, j] = (c0[i + 1, j] + c0[i - 1, j] + c0[i, jp] + c0[i, jm]) / 4
        delta = np.max(np.abs(c1 - c0))
        deltas.append(delta)
        print(delta)

        c0[1:N] = c1[1:N]
    return c1, deltas

def gauss_seidel(c0, c1, delta):
    deltas = []
    while (delta >= tolerance):
        for i in range(1, N):
            for j in range(N + 1):
                jp = j + 1 if j < N else 0    # periodic            conditions
                jm = j - 1 if j > 0 else N    #           boundary
                c1[i, j] = (c0[i + 1, j] + c1[i - 1, j] + c0[i, jp] + c1[i, jm]) / 4
        delta = np.max(np.abs(c1 - c0))
        print(delta)
        deltas.append(delta)

        c0[1:N] = c1[1:N]
    return c1, deltas

def sor(c0, c1, delta, omega):
    deltas = []
    while (delta >= tolerance):
        for i in range(1, N):
            for j in range(N + 1):
                jp = j + 1 if j < N else 0    # periodic            conditions
                jm = j - 1 if j > 0 else N    #           boundary
                c1[i, j] = ((1 - omega) * c0[i, j] +
                            omega / 4 * (c0[i + 1, j] + c1[i - 1, j] + c0[i, jp] + c1[i, jm]))

        delta = np.max(np.abs(c1 - c0))
        deltas.append(delta)
        print(delta)

        c0[1:N] = c1[1:N]
    return c1, deltas


jacob = jacobi(np.copy(c0), np.copy(c1), delta)
gauss = gauss_seidel(np.copy(c0), np.copy(c1), delta)

def plot_diffusion():
    sorry = sor(np.copy(c0), np.copy(c1), delta, 1.85)
    plt.title("Diffusion cross section")
    plt.xlabel(r"$y$")
    plt.ylabel("Concentration")
    plt.plot(xs, [1 - x for x in xs], label="analytic")
    plt.plot(xs, jacob[0][:, N // 2], label="Jacobi")
    plt.plot(xs, gauss[0][:, N // 2], label="Gauss-Seidel")
    plt.plot(xs, sorry[0][:, N // 2], label=r"SOR $\omega=1.85$")
    plt.legend()
    plt.savefig("time_ind", dpi=400)

def plot_convergence():
    sor1 = sor(np.copy(c0), np.copy(c1), delta, 1.7)
    sor2 = sor(np.copy(c0), np.copy(c1), delta, 1.8)
    sor3 = sor(np.copy(c0), np.copy(c1), delta, 1.891)
    for rs in ((jacob, "Jacobi"), (gauss, "Gauss-Seidel"),
               (sor1, r"SOR, $\omega=1.7$"), (sor2, r"SOR, $\omega=1.8$"),
               (sor3, r"SOR, $\omega=1.891$")):
        r, label = rs
        c, deltas = r
        plt.loglog(range(len(deltas)), deltas, label=label)
    plt.legend()
    plt.title("Convergence vs. iterations")
    plt.xlabel(r"iterations ($k$)")
    plt.ylabel(r"Convergence measure ($\delta$)")
    plt.savefig("time_ind_conv", dpi=400)
    plt.show()


# finding optimal omega
# range of omegas is manually modified to find optimal omega

# omegas = np.arange(1.7, 2, 0.01)
# sors = [sor(np.copy(c0), np.copy(c1), delta, omega) for omega in omegas]

# we check the iterations with [len(x[1] for x in sors] and change the
# omegas range to further pinpoint the optimum
optimega = 1.891                # N = 50, no objects

def sor_object(c0, c1, delta, omega, obj):
    deltas = []
    while (delta >= tolerance):
        for i in range(1, N):
            for j in range(N + 1):
                jp = j + 1 if j < N else 0    # periodic            conditions
                jm = j - 1 if j > 0 else N    #           boundary
                if obj[i, j] == 1:
                    c1[i, j] = 0  # object
                else:
                    c1[i, j] = ((1 - omega) * c0[i, j] +
                                omega / 4 * (c0[i + 1, j] + c1[i - 1, j] + c0[i, jp] + c1[i, jm]))

        delta = np.max(np.abs(c1 - c0))
        deltas.append(delta)
        print(delta)

        c0[1:N] = c1[1:N]
    return c1, deltas


# objects in the domain for N = 50
# a 10 x 10 square in the middle of the domain
# 1 is the object, 0 is nothing
c_square = np.zeros((N + 1, N + 1))
square_row = np.array(21 * [0] + 10 * [1] + 20 * [0])
c_square[20:31] = square_row

c_zero = np.zeros((N + 1, N + 1))

c_two_rectangle = np.zeros((N + 1, N + 1))
c_two_rectangle[5:7] = square_row
c_two_rectangle[45:7] = square_row

obj1 = sor_object(np.copy(c0), np.copy(c1), delta, optimega, c_square)
obj2 = sor_object(np.copy(c0), np.copy(c1), delta, optimega, c_two_rectangle)
