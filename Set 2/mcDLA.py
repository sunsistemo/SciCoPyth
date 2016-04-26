import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import erfc, erf, sqrt, log10
from random import random, randint, choice
from collections import namedtuple
from tqdm import trange


latSize = 64 + 2               # lattice + padding
growthPar = .5
lattice = np.zeros((latSize, latSize))
# x is i is VERTICAL and y is j is HORIZONTAL

class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Walker:
    def __init__(self, lattice, x, y):
        self.lattice = lattice
        self.pos = Position(x, y)
        self.alive = True

    def step(self):
        dx, dy = choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        self.pos = Position(self.pos.x + dx, self.pos.y + dy)

        if self.pos.x > len(self.lattice) - 2 or self.pos.x < 1:
            self.die()

        if self.pos.y > len(self.lattice[0]) - 2:
            self.pos.y = 1
        elif self.pos.y < 1:
            self.pos.y = len(self.lattice[0]) - 2

    def check_friends(self):
        i, j = self.pos.x, self.pos.y
        friends = self.lattice[i-1,j] + self.lattice[i+1,j] + self.lattice[i,j-1] + self.lattice[i,j+1]
        return (friends > 0)

    def merge(self):
        self.lattice[self.pos.x, self.pos.y] = 1
        self.die()
        return self.lattice

    def die(self):
        self.alive = False


def grow_step(lattice):
    walker = Walker(lattice, randint(0, len(lattice[0]) - 1), latSize//2)
    while walker.alive:
        if walker.check_friends():
            lattice = walker.merge()
        walker.step()
    return lattice


def grow(lattice, steps):
    for n in range(steps):
        lattice = grow_step(lattice)
    return lattice

#################
# VISUALIZATION #
#################
fig, ax = plt.subplots()
hMap = ax.imshow(lattice)

def animate(frame_number):
    global lattice
    # lattice = grow_ani_step(lattice)
    lattice = next(ani_gen)
    hMap.set_array(lattice)
    return hMap,

def grow_ani_step(lattice):
    while True:
        walker = Walker(lattice, randint(0, len(lattice[0]) - 1), latSize//2)
        while walker.alive:
            if walker.check_friends():
                lattice = walker.merge()
            walker.step()
            yield lattice
        yield lattice


def start_animation():
    ani = animation.FuncAnimation(fig, animate, range(1, 1000), init_func=init, interval=1, blit=True)
    plt.colorbar(hMap)
    plt.show()

def init():
    hMap.set_array(np.ma.array(lattice))
    return hMap,

ani_gen = grow_ani_step(lattice)

# lattice = grow(lattice, 1000)
# plt.imshow(lattice)
# plt.show()

start_animation()
