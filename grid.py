import numpy as np
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