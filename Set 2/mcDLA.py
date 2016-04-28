import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import erfc, erf, sqrt, log10
from random import random, randint, choice
from collections import namedtuple
from tqdm import trange


latSize = 64 + 2               # lattice + padding
p_stick = .1
lattice = np.zeros((latSize, latSize))
# x is i is VERTICAL and y is j is HORIZONTAL
walkerLattice = np.copy(lattice)
lattice[latSize-2, latSize//2] = 1

class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Walker:
    def __init__(self, lattice, x, y):
        self.lattice = lattice
        self.pos = Position(x, y)
        self.alive = True

    def step(self, stepList = [(1, 0), (-1, 0), (0, 1), (0, -1)]):
        dx, dy = choice(stepList)
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

    def merge(self, stickProb):
        if random() < stickProb:
            self.lattice[self.pos.x, self.pos.y] = 1
            self.die()
            return self.lattice
        else:
            self.no_merge()
            return self.lattice

    def no_merge(self):
        choiceList = [nb for nb in [(1, 0), (-1, 0), (0, 1), (0, -1)]
                      if lattice[self.pos.x + nb[0], self.pos.y + nb[1]] == 0]
        if len(choiceList) < 0:
            self.step(choiceList)
        else:
            self.die()

    def die(self):
        self.alive = False


def grow_step(lattice):
    walker = Walker(lattice, 1, randint(1, len(lattice[0]) - 1))
    while walker.alive:
        if walker.check_friends():
            lattice = walker.merge(p_stick)
        walker.step()
    return lattice


def grow(lattice, steps):
    for n in range(steps):
        lattice = grow_step(lattice)
    return lattice

#################
# VISUALIZATION #
#################
walker = Walker(lattice, 1, randint(1, len(lattice[0]) - 2))
walkerLattice = np.copy(lattice)
walkerLattice[walker.pos.x, walker.pos.y] = 1

fig, ax = plt.subplots()
hMap = ax.imshow(walkerLattice)

def animate(frame_number):
    global lattice

    # lattice = grow_ani_step(lattice)
    lattice = next(ani_gen)
    hMap.set_array(lattice)
    return hMap,

def grow_ani_step(lattice, aniWalk):
    while True:
        walker = Walker(lattice, 1, randint(1, len(lattice[0]) - 2))
        # print("new walker at", walker.pos.x, walker.pos.y)
        while walker.alive:
            if walker.check_friends():
                lattice = walker.merge(p_stick)
                walkerLattice = np.copy(lattice)
                yield walkerLattice

            walker.step()
            if aniWalk:
                walkerLattice = np.copy(lattice)
                try:
                    walkerLattice[walker.pos.x, walker.pos.y] = 1
                except:
                    print(walker.pos)
                yield walkerLattice
        # yield walkerLattice


def start_animation():
    ani = animation.FuncAnimation(fig, animate, range(1, 1000), init_func=init, interval=1, blit=True)
    plt.colorbar(hMap)
    plt.show()

def init():
    hMap.set_array(np.ma.array(walkerLattice))
    return hMap,

ani_gen = grow_ani_step(lattice, False)
# lattice = grow(lattice, 1000)
# plt.imshow(lattice)
# plt.show()

start_animation()
print("done")
