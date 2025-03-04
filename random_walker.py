import random
import matplotlib.pyplot as plt
import numpy as np

class Walker():
    def __init__(self, grid, sticking_probability):
        self.active = True
        self.growth_candidate = False
        self.location = [random.randint(1, grid.N), grid.N]
        self.stick_prob = sticking_probability
        self.movements = []
        self.movements.append(self.location[:])

    def walk(self, grid):
        
        # if direction is 0, walker moves in x-direction.
        # if direction is 1, walker moves in y-direction.
        direction = random.choice([0, 1])
        posneg = random.choice([-1, 1])

        while ([self.location[0] + posneg, self.location[1]] in grid.cluster) or ([self.location[0], self.location[1] + posneg] in grid.cluster):
            direction = random.choice([0, 1])
            posneg = random.choice([-1, 1])

        self.location[direction] += posneg

        if self.location[1] < 1 or self.location[1] > grid.N:
            self.active = False
            return

        if self.location[0] < 1:
            self.location[0] = grid.N
        elif self.location[0] > grid.N:
            self.location[0] = 1

        self.movements.append(self.location[:])

        return self.location

    def check_cluster(self, grid):
        for growth_candidate in grid.cluster:
            if (abs(self.location[0] - growth_candidate[0]) + abs(self.location[1] - growth_candidate[1])) == 1 and self.active == True and random.random() <= self.stick_prob:
                self.growth_candidate = True
                self.active = False
                grid.cluster.append(self.location)
                return True
        return False
    
    def visualise_walk(self, grid):
        print(self.movements)
        x, y = [], []

        for movement in self.movements:
            x.append(movement[0])
            y.append(movement[1])
        print(x, y)
        plt.plot(x, y, "k-")
        plt.axis("square")
        plt.grid(True)
        plt.xticks([x for x in np.arange(0,grid.N+2)])
        plt.yticks([y for y in np.arange(0,grid.N+2)])
        plt.show()
