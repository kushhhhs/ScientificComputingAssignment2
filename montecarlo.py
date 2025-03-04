# for B, just put sticking_probability = 1

from grid import Grid
from random_walker import Walker

n = 100
number_walkers = 10000
sticking_probability = 1
grid = Grid(n)

for i in range(number_walkers):
    random_walker = Walker(grid, sticking_probability)
    while random_walker.active:
        random_walker.walk(grid)
        random_walker.check_cluster(grid)
    # random_walker.visualise_walk(grid)

grid.visualise()
print(len(grid.cluster))

cluster_sizes = {}

# experiment to see the effect of different sticking probabilities on size of cluster

for sticking_probability in [0.5, 1]:
    cluster_sizes[sticking_probability] = []
    for _ in range(10):
        grid = Grid(n)
        for i in range(number_walkers):
            random_walker = Walker(grid, sticking_probability)
            while random_walker.active:
                random_walker.walk(grid)
                random_walker.check_cluster(grid)
        cluster_sizes[sticking_probability].append(len(grid.cluster))
        print(cluster_sizes)

print(cluster_sizes)
