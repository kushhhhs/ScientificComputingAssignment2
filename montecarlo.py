# for B, just put sticking_probability = 1

from grid import Grid
from random_walker import Walker

n = 100
number_walkers = 3000
sticking_probability = 0.6
grid = Grid(n)

for i in range(number_walkers):
    random_walker = Walker(grid, sticking_probability)
    while random_walker.active:
        random_walker.walk(grid)
        random_walker.check_cluster(grid)

grid.visualise()
