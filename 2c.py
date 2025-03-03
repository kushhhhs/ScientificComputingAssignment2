"""This module answers assignment 2.3 and will later likely be divided 
into seperate modules"""

import numpy as np

def laplacian_roll(grid, dx):
    """Computes the laplacian by implementing np.roll, implements 
    periodic boundary conditions"""
    down = np.roll(grid, 1, axis=0)
    up = np.roll(grid, -1, axis=0)
    right = np.roll(grid, 1, axis=1)
    left = np.roll(grid, -1, axis=1)

    return ((down + up + right + left - 4*grid) / dx**2)

def boundary_neumann(grid):
    """Implements the Neumann Boundary Conditions"""
    f = np.pad(grid, 1)
    #print(f)

    # Periodic conditions
    #f[:, 0] = f[:, -2]
    #f[:, -1] = f[:, 1]
    #f[0, :] = f[-2, :]
    #f[-1, :] = f[1, :]

    return f

def gray_scott():
    """Calculates The gray scott reaction-diffusion equations."""
    return

def update():
    """Updates the grid for U and V (seperate grids)"""

def plot():
    """plots concentrations U and V
    - For several parameter choices
    - Multiple graphs below for different parameter changes?
    - Can find sets for specific parameter settings online
        - Change those and note trends/changes in structure/ effects"""

# Optional
def implement_mask():
    """Implement a mask
    - For which parameters?"""

grid = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]])
grid = boundary_neumann(grid)
print(grid)

#print(grid[:, 0])
#print(grid[:, -2])