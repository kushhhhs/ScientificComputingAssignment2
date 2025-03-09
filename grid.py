import numpy as np
import matplotlib.pyplot as plt

from random_walker import Walker

class Grid():
    """
    This class-object creates a domain-grid and cluster to investigate the use
    of the Monte Carlo method for DLA.
    """

    def __init__(
            self,
            grid_size: int,
            sticking_probability: float,
            size_cluster: int):
        """
        Initialises the Grid class-object and creates initial cluster
        containing a single particle in the middle at the bottom of the domain.

        Parameters:
        - grid_size (int): Size of the grid (grid_size x grid_size).
        - stick_prob (float): The sticking probability of each particle to the
          cluster (0 < p <= 1).
        - size_cluster (int): The desired size of the cluster.
        """
        self.grid_size: int = grid_size
        self.stick_prob: float = sticking_probability
        self.size_cluster: int = size_cluster
        self.cluster: list[list[int]] = [[int(grid_size/2), 1]]

    def create_cluster(self):
        """
        Creates Walker class-objects that act as particles behaving with
        Brownian Motion until the desired size of the cluster is reached.
        """
        while self.check_size() < self.size_cluster:
            random_walker = Walker(self)
            while random_walker.active:
                random_walker.walk(self)
                random_walker.check_cluster(self)
    
    def check_size(self):
        """
        Returns the number of aggregated particles of the cluster.
        """
        return len(self.cluster)

    def visualise(self):
        """
        Plots a visual representation of the cluster on the domain.
        """
        for aggre_part in self.cluster:
            plt.plot(aggre_part[0], aggre_part[1], "ro", markersize = 1)
        plt.axis("square")
        plt.xlim(0,101)
        plt.ylim(0,101)
        plt.title(rf"Monte Carlo DLA cluster $p_s={self.stick_prob}$")
        plt.show()

    def print_stats(self):
        """
        Prints some information about the cluster.
        """
        print(f"size: {len(self.cluster)}")
        max_y = 0
        min_x, max_x = int(self.grid_size/2), int(self.grid_size/2)
        for aggre_part in self.cluster:
            if aggre_part[0] < min_x: min_x = aggre_part[0]
            if aggre_part[0] > max_x: max_x = aggre_part[0]
            if aggre_part[1] > max_y: max_y = aggre_part[1]
        print(f"width: {max_x - min_x}")
        print(f"height: {max_y - 1}")

    def check_spread(self):
        """
        Returns the height and width of the cluster.
        """
        max_y, min_x, max_x = 1, int(self.grid_size/2), int(self.grid_size/2)
        for aggre_part in self.cluster:
            if aggre_part[0] < min_x: min_x = aggre_part[0]
            if aggre_part[0] > max_x: max_x = aggre_part[0]
            if aggre_part[1] > max_y: max_y = aggre_part[1]
        return max_y - 1, max_x - min_x