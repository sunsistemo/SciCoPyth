# -*- coding: utf-8 -*-
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

L = 1
c = 1
N = 50
dx = L / N
# dt = 0.01
dt = (dx ** 2 / (2 * c)) * 0.9

x = [i * dx for i in range(N + 1)]
u0 = [0] * (N + 1)                    # ψ(x, t = 0)
v0 = [0] * (N + 1)                    # ψ'(x, t = 0) = 0
num_steps = 100

assert dt <= (dx ** 2 / (2 * c)), "Simulation is unstable, aborting!"

fig, ax = plt.subplots()
ax = plt.axes(xlim=(0, L), ylim=(-1.5, 1.5))
line, = ax.plot(x, u0)

def init():
    line.set_xdata(x)
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

def animate(frame_number):
    for i in range(1, N):     # at i = 0, N  we have ψ = 0 (boundary condition)
        u2[i] = 2 * u1[i] - u0[i] + c * (dt / dx) ** 2 * (u1[i + 1] - 2 * u1[i] + u1[i - 1])

    for i in range(1, N):
        u0[i] = u1[i]
        u1[i] = u2[i]
    line.set_ydata(u2)  # update the ani data
    return line,

# initial condition
for i in range(1, N):
    xi = i * dx
    # u0[i] = np.sin(2 * np.pi * xi)
    u0[i] = np.sin(5 * np.pi * xi)
    # u0[i] = np.sin(5 * np.pi * xi) if 1 / 5 < xi < 2 / 5 else 0

u1 = [0] * (N + 1)
for i in range(N):
    u1[i] = u0[i] + dt * v0[i]

u2 = [0] * (N + 1)

def start_animation():
    ani = animation.FuncAnimation(fig, animate, range(1, num_steps), init_func=init,
                              interval=1, blit=True)
    plt.show()

def plot():
    N = 50
    step = lambda n: [animate(1) for _ in range(int(n))]
    second = 1 / dt
    plt.title("Wave equation solutions: dt=%.5f, dx=%.2f" % (dt, dx))
    plt.xlabel("x")
    plt.ylabel(r"$\psi$")
    step(0.01 * second)
    plt.plot(x, u2, label="t=0.01")
    step(0.06 * second)
    plt.plot(x, u2, label="t=0.07")
    step(0.03 * second)
    plt.plot(x, u2, label="t=0.1")
    step(0.1 * second)
    plt.plot(x, u2, label="t=0.2")
    plt.legend(ncol=4)
    plt.savefig("wave_eq", dpi=400)

start_animation()
