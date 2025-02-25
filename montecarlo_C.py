# "The walker is however not allowed to move into a site belonging to the cluster."

# If the walker is about to move into a sire belonging to the cluster, should
# it just not move and go through the probability again or should it be a while
# loop so the walker is forced to move before it gets the chance to become part
# of the cluster again?

import numpy as np
import random
import matplotlib.pyplot as plt

class Grid():
    def __init__(self, N):
        self.N = N

        # i assume the initial cluster is a single candidate ~ in the middle?
        self.cluster = [[int(N/2), int(N/2)]]

    def visualise(self):
        for growth_candidate in self.cluster:
            plt.plot(growth_candidate[0], growth_candidate[1], "o", markersize = 5)
        plt.axis("square")
        plt.grid(True)
        plt.xticks([x for x in np.arange(0,self.N+2)])
        plt.yticks([y for y in np.arange(0,self.N+2)])
        plt.show()

class Walker():
    def __init__(self, grid):
        self.active = True
        self.growth_candidate = False
        self.location = [random.randint(1, grid.N), grid.N]

    def walk(self, grid):
        
        # if direction is 0, walker moves in x-direction.
        # if direction is 1, walker moves in y-direction.
        direction = random.choice([0, 1])
        posneg = random.choice([-1, 1])

        if [self.location[0] + posneg, self.location[1]] in grid.cluster:
                return
        else:
            if [self.location[0], self.location[1] + posneg] in grid.cluster:
                return

        self.location[direction] += posneg

        if self.location[1] < 1 or self.location[1] > grid.N:
            self.active = False
            return

        if self.location[0] < 1:
            self.location[0] = grid.N
        elif self.location[0] > grid.N:
            self.location[0] = 1

        return self.location

    def check_cluster(self, grid):
        for growth_candidate in grid.cluster:
            if (abs(self.location[0] - growth_candidate[0]) + abs(self.location[1] - growth_candidate[1])) == 1 and self.active == True:
                self.growth_candidate = True
                self.active = False
                grid.cluster.append(self.location)
                return True
        return False

n = 100
number_walkers = 2000
grid = Grid(n)

for i in range(number_walkers):
    random_walker = Walker(grid)
    while random_walker.active:
        random_walker.walk(grid)
        random_walker.check_cluster(grid)

grid.visualise()
