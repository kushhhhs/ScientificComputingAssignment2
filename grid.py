import numpy as np
import matplotlib.pyplot as plt

from random_walker import Walker

class Grid():
    def __init__(self, grid_size, sticking_probability, size_cluster):
        self.grid_size = grid_size
        self.stick_prob = sticking_probability
        self.size_cluster = size_cluster

        # i assume the initial cluster is a single candidate ~ in the middle at the bottom?
        self.cluster = [[int(grid_size/2), 1]]

    def create_cluster(self):
        while self.check_size() < self.size_cluster:
            random_walker = Walker(self)
            while random_walker.active:
                random_walker.walk(self)
                random_walker.check_cluster(self)
        return self
    
    def check_size(self):
        return len(self.cluster)

    def visualise(self):
        for growth_candidate in self.cluster:
            plt.plot(growth_candidate[0], growth_candidate[1], "ro", markersize = 2)
        plt.axis("square")
        plt.xlim(0,101)
        plt.ylim(0,101)
        plt.title(rf"Monte Carlo DLA cluster $p_s={self.stick_prob}$")
        plt.show()

    def print_stats(self):
        print(f"size: {len(self.cluster)}")
        max_y = 0
        min_x, max_x = int(self.grid_size/2), int(self.grid_size/2)
        for growth_candidate in self.cluster:
            if growth_candidate[0] < min_x: min_x = growth_candidate[0]
            if growth_candidate[0] > max_x: max_x = growth_candidate[0]
            if growth_candidate[1] > max_y: max_y = growth_candidate[1]
        print(f"width: {max_x - min_x}")
        print(f"height: {max_y - 1}")