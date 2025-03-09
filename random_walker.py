import random
import matplotlib.pyplot as plt
import numpy as np

class Walker():
    """
    This class-object creates a particle that behaves as a random walker with
    periodic boundary conditions on the left and right border and regeneration
    on the top and bottom borders.
    """

    def __init__(self, grid):
        """
        Initialises the Walker class-object by giving it an initial random
        location on the top border.

        Parameters:
        - grid (Grid): Uses grid.grid_size to put the initial location within
          the boundaries of the domain.
        """
        self.active: bool = True
        self.clustered: bool = False
        self.location: list[int] = [random.randint(1, grid.grid_size), grid.grid_size]
        self.movements: list[list[int]] = [self.location[:]]

    def walk(self, grid):
        """
        The walker object takes a step in any direction.

        Parameters:
        - grid (Grid): Uses grid.grid_size to check whether walker moves into a
          border and uses grid.cluster to check whether walker wishes to move
          into the cluster (possible when stick_prob < 1).
        """

        # if direction is 0, walker moves in x-direction.
        # if direction is 1, walker moves in y-direction.
        direction = random.choice([0, 1])
        posneg = random.choice([-1, 1])

        while ([self.location[0] + posneg, self.location[1]] in grid.cluster) or ([self.location[0], self.location[1] + posneg] in grid.cluster):
            direction = random.choice([0, 1])
            posneg = random.choice([-1, 1])

        self.location[direction] += posneg

        if self.location[1] < 1 or self.location[1] > grid.grid_size:
            self.active = False
            return

        if self.location[0] < 1:
            self.location[0] = grid.grid_size
        elif self.location[0] > grid.grid_size:
            self.location[0] = 1

        self.movements.append(self.location[:])

        return self.location

    def check_cluster(self, grid):
        """
        If Walker object is adjacent to cluster, becomes part of cluster with
        likelihood of sticking probability.

        Parameters:
        - grid (Grid): Uses grid.stick_prob and grid.cluster to add particle to
          list of aggregated particles.

        Returns True if Walker becomes part of cluster and returns False
        otherwise.
        """
        for aggre_part in grid.cluster:
            if (abs(self.location[0] - aggre_part[0]) + abs(self.location[1] - aggre_part[1])) == 1 and self.active == True and random.random() <= grid.stick_prob:
                self.clustered = True
                self.active = False
                grid.cluster.append(self.location)
                return True
        return False
    
    def visualise_walk(self, grid):
        """
        Plots path of particle until it becomes inactive (either due to moving
        into top or bottom boundary or due to becoming part of cluster) as a
        check of the path.
        """
        print(self.movements)
        x, y = [], []

        for movement in self.movements:
            x.append(movement[0])
            y.append(movement[1])
        print(x, y)
        plt.plot(x, y, "k-")
        plt.axis("square")
        plt.grid(True)
        plt.xticks([x for x in np.arange(0,grid.grid_size+2)])
        plt.yticks([y for y in np.arange(0,grid.grid_size+2)])
        plt.show()
