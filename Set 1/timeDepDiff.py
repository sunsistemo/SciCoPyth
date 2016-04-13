import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from math import erfc, erf, sqrt, log10

L = 1.0
N = 30
dx= L/N
dy = L/N
y0 = 0
yL = 1
nsteps = 1000
D = 1
dt = 0.9 * dx**2 /(D * 4)

c0 = np.zeros((N + 1, N + 1))
c0[0] = 1
c1 = np.copy(c0)
fig, ax = plt.subplots()
hMap = ax.imshow(c0)

def diff_step():
    for i in range(1, N):
        for j in range(N + 1):
            jp = j + 1 if j < N else 0
            jm = j - 1 if j > 0 else N
            c1[i,j] = c0[i,j] + (dt * D/(dx**2)) * (c0[i-1, j] + c0[i+1, j] + c0[i, jm] + c0[i, jp] - 4 * c0[i, j])
    c0[:] = c1[:]

def ana_sol(t, bins):
    sol = np.zeros(bins+1)
    db = L/bins
    if t == 0.0:
        sol[0] = 1
    else:
        for b in range(0, bins+1):
            sol[bins - b] = sum(erfc((1 - b*db + 2*i)/(2*sqrt(D*t))) - erfc((1 + b*db + 2*i)/(2*sqrt(D*t))) for i in range(10000))
    return sol

def init():
    hMap.set_array(np.ma.array(c0))
    return hMap,

def animate(frame_number):
# for num in range(nsteps):
    diff_step()
    hMap.set_array(c1)
    return hMap,

def start_animation():
    ani = animation.FuncAnimation(fig, animate, range(1, nsteps), init_func=init, interval=1, blit=True)
    plt.colorbar(hMap)
    plt.show()

def generate_2Dplots():
    times = np.array([0.0001, 0.001, 0.01, 0.1, 1])
    nsteps = times / dt

    steps = 0
    for t in times:
        steps = int(t/dt - steps)
        print(steps)
        for n in range(steps):
            diff_step()

        plt.cla()
        plt.clf()
        plt.imshow(c1)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar()
        plt.savefig("diffusion_%d.png" % int(5+log10(t)))

def generate_1Dplot():
    times = np.array([0, 0.001, 0.01, 0.1, 1])
    nsteps = times / dt
    steps = 0
    plt.close("all")
    for t in times:
        steps = int(t/dt - steps)
        print(t)
        for n in range(steps):
            diff_step()
        xn = np.linspace(0,L,N+1)
        plt.plot(xn,c1[:,0], label="Numerical t=%.3f" % t)
        bins = 100
        xa = np.linspace(0,L,bins+1)
        plt.plot(xa,ana_sol(float(t), bins), "k--")
    plt.xlabel("y")
    plt.ylabel("c(y)")
    plt.legend()
    plt.savefig("diffusion1D.png")
    plt.show()

# print(ana_sol(.1))

# generate_1Dplot()
# generate_2Dplots()
start_animation()
