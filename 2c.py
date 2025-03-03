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


def boundary_periodic(grid):
    return


def gray_scott(dU, dV, LU, LV, U, V, feed, kill):
    """Calculates The gray scott reaction-diffusion equations.
    LU and LV: Laplacian of U and V
    dU and dV: respective diffusion coefficients"""
    du_dt = (Du * LU) - (U * V**2) + feed(1 - U)
    dv_dt = (Dv * LV) + (U * V**2) - V(feed + kill)

    return du_dt, dv_dt


def update(grid, dx):
    """Updates the grid for U and V (seperate grids)"""
    
    # Compute laplacian
    laplacian_roll(grid, dx)

    #GS recation

    # Update U and V

    # apply Neumann?


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


def init(n, center):
    """Initializes the GS model"""

    U = np.zeros((n, n), dtype=np.float64)
    V = np.zeros((n, n), dtype=np.float64)

    dx = np.float64(1.0)
    dt = np.float64(1.0)

    dU = np.float64(0.16)
    dV = np.float64(0.08)

    feed = np.float64(0.035)
    kill = np.float64(0.06)

    # Initializes all U values at 0.5
    U[:,:] = np.float64(0.5)
    #V[]
    print(U)


def get_center_positions(n, c):
    """Gets the center positions for the grid square mask"""
    start = (n - c) // 2
    end = start + c
    center_positions = []
    
    for i in range(start, end):
        for j in range(start, end):
            center_positions.append([i, j])

    return center_positions

n = 8
c = 2
get_center_positions(n, c)
# grid = np.array([[1, 2, 3, 4],
#                 [5, 6, 7, 8],
#                 [9, 10, 11, 12],
#                 [13, 14, 15, 16]])
# grid = boundary_neumann(grid)
# print(grid)
