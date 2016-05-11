import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import trange
from numba import jit


N = 300 + 1
dt = 1
dx = dy = 1

Du = 0.16
Dv = 0.08
f = 0.035
k = 0.060

# parameters 2
# f = 0.03
# k = 0.055

u0, v0 = 0.5, 0.25
u = np.zeros((N, N))
v = np.zeros((N, N))
u[:] = u0

M = N // 11
square_slice = slice(N // 2 - M, N // 2 + M)
r = np.zeros((N))
r[square_slice] = v0
v[square_slice] = r

# add noise
u += 0.05 * np.random.random((N, N))
v += 0.05 * np.random.random((N, N))

@jit
def step(u, v):
    u0 = np.copy(u)
    v0 = np.copy(v)
    for i in range(N):
        for j in range(N):
            # periodic boundary conditions at all edges
            # ip is i + 1, jm is j - 1 etc.
            ip = i + 1 if i < N - 1 else 0
            im = i - 1 if i > 0 else N - 1
            jp = j + 1 if j < N - 1 else 0
            jm = j - 1 if j > 0 else N - 1
            u[i, j] = u0[i, j] + dt * (Du / dx ** 2 * (u0[ip, j] + u0[im, j] + u0[i, jp] + u0[i, jm] - 4 * u0[i, j]) - u0[i, j] * v0[i, j] ** 2 + f * (1 - u0[i, j]))
            v[i, j] = v0[i, j] + dt * (Dv / dx ** 2 * (v0[ip, j] + v0[im, j] + v0[i, jp] + v0[i, jm] - 4 * v0[i, j]) + u0[i, j] * v0[i, j] ** 2 - (f + k) * v0[i, j])
    return u, v


def init():
    u_im.set_array(np.ma.array(u))
    v_im.set_array(np.ma.array(v))
    return u_im, v_im


def animate(frame_number):
    global u, v
    u, v = step(u, v)
    u_im.set_array(u)
    v_im.set_array(v)
    print(frame_number)
    return u_im, v_im

def start_animation():
    ani = animation.FuncAnimation(fig, animate, None, init_func=init, interval=10, blit=True)
    plt.show()


def plot(steps):
    global u, v

    for i in trange(steps):
        u, v = step(u, v)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))
    u_im = ax1.imshow(u)
    v_im = ax2.imshow(v)
    ax1.set_title(r"$u$")
    ax2.set_title(r"$v$")
    ax1.set_xlabel(r"$x$")
    ax2.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax2.set_ylabel(r"$y$")
    fig.colorbar(u_im, ax=ax1)
    fig.colorbar(v_im, ax=ax2)
    plt.show()
    plt.savefig("gs2_%d" % steps)



# animation
fig, (ax1, ax2) = plt.subplots(1, 2)
u_im = ax1.imshow(u)
v_im = ax2.imshow(v)

start_animation()
# plot(50000)
