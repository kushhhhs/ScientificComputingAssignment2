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
            plt.xlim(0, self.N-1)
            plt.ylim(0, self.N-1)
            plt.axis("square")
            plt.grid(True)
            plt.xticks([x for x in range(0,N)])
            plt.yticks([y for y in range(0,N)])
            plt.plot(growth_candidate[1], growth_candidate[0], "ko")
        plt.show()

class Walker():
    def __init__(self, grid):
        self.active = True
        self.growth_candidate = False
        self.location = [0, random.randint(0, grid.N-1)]

    def walk(self, grid):
        
        # if direction is 0, walker goes up/down. if direction is 1, walker goes left/right
        direction = random.choice([0, 1])
        posneg = random.choice([-1, 1])
        self.location[direction] += posneg
        if self.location[0] < 0 or self.location[0] > grid.N-1:
            self.active = False
            return
        
        if self.location[1] < 0:
            self.location[1] = grid.N-1
        elif self.location[1] > grid.N-1:
            self.location[1] = 0

        return self.location

    def check_cluster(self, grid):
        for growth_candidate in grid.cluster:
            if (abs(self.location[0] - growth_candidate[0]) == 1 and self.location[1] == growth_candidate[1]) or (abs(self.location[1] - growth_candidate[1]) == 1 and self.location[0] == growth_candidate[0]):
                self.growth_candidate = True
                self.active = False
                grid.cluster.append(self.location)
                return True
        return False

N = 21
number_walkers = 200
grid = Grid(N)

for i in range(number_walkers):
    random_walker = Walker(grid)
    while random_walker.active:
        random_walker.walk(grid)
        random_walker.check_cluster(grid)

grid.visualise()
