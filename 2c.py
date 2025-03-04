"""This module answers assignment 2.3 and will later likely be divided 
into seperate modules"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def laplacian_roll(grid, dx):
    """Computes the laplacian by implementing np.roll, implements 
    periodic boundary conditions"""
    down = np.roll(grid, 1, axis=0)
    up = np.roll(grid, -1, axis=0)
    right = np.roll(grid, 1, axis=1)
    left = np.roll(grid, -1, axis=1)

    return ((down + up + right + left - 4*grid) / dx**2)


def boundary_periodic(grid):
    """Enforces periodic boundary conditions"""
    grid[0, :] = grid[-2, :]
    grid[-1, :] = grid[1, :]
    grid[:, 0] = grid[:, -2]
    grid[:, -1] = grid[:, 1]
    return grid


def boundary_neumann(grid):
    """Implements the Neumann Boundary Conditions
    e.g. the derivative at the boundary is zero (no flux across boundary)"""
    f = np.pad(grid, 1)
    #print(f)

    f[:, 0] = f[:, 1]
    f[:, -1] = f[:, -2]
    f[0, :] = f[1, :]
    f[-1, :] = f[-2, :]

    return f

def laplacian_neumann(grid, dx):
    """Computes Laplacian for von Neumann BC"""
    top = grid[:-2, 1:-1]
    bottom = grid[2:, 1:-1]
    left = grid[1:-1, :-2]
    right = grid[1:-1, 2:]
    
    return ((top + bottom + left + right - 4*grid[1:-1, 1:-1]) / dx**2)



def gray_scott(dU, dV, LU, LV, U, V, feed, kill):
    """Calculates The gray scott reaction-diffusion equations.
    LU and LV: Laplacian of U and V
    dU and dV: respective diffusion coefficients"""
    dUdt = (dU * LU) - (U * V**2) + feed*(1 - U)
    dVdt = (dV * LV) + (U * V**2) - V*(feed + kill)

    return dUdt, dVdt


def update(U, V, dx, dt, dU, dV, kill=0.06, feed=0.035):
    """Updates the grid for U and V (seperate grids)"""
    
    # Compute laplacian for U and V
    LU = laplacian_neumann(U, dx)
    LV = laplacian_neumann(V, dx)
    
    # GS reaction
    dUdt, dVdt = gray_scott(dU, dV, LU, LV, U[1:-1, 1:-1], V[1:-1, 1:-1], feed, kill)

    # Update U and V
    U[1:-1, 1:-1] += dt * dUdt
    V[1:-1, 1:-1] += dt * dVdt

    # apply Neumann?
    U_new = boundary_neumann(U[1:-1, 1:-1])
    V_new = boundary_neumann(V[1:-1, 1:-1])
    
    return U_new, V_new


def plot():
    """plots concentrations U and V
    - For several parameter choices
    - Multiple graphs below for different parameter changes?
    - Can find sets for specific parameter settings online
        - Change those and note trends/changes in structure/ effects"""


def init_neumann(n, center, i_value_U=0.5, i_value_V=0.25):
    """Initializes the GS model"""

    U = np.zeros((n, n))
    V = np.zeros((n, n))

    # Initializes all U values at 0.5
    U[:,:] = i_value_U

    # Sets value of V to center positions
    V = center_positions(n, center, V, i_value_V)

    # Apply von Neuman
    U = boundary_neumann(U)
    V = boundary_neumann(V)

    return U, V
    

def center_positions(n, c, V, i_value_V):
    """Gets the center positions for the grid square mask"""
    start = (n - c) // 2
    end = start + c
    
    for i in range(start, end):
        for j in range(start, end):
           V[i, j] = i_value_V

    return V



def plot_field(field):
    plt.figure(figsize=(6, 6))
    # Display the interior part of the field (excluding ghost cells)
    plt.imshow(field[1:-1, 1:-1], cmap="jet", interpolation='nearest')
    plt.colorbar()
    plt.show()


def create_animation(n, center, dx=1.0, dt=1.0, dU=0.16, dV=0.08, feed=0.035, kill=0.06, i_value_U=0.5, i_value_V=0.25):

    # Initialize the padded fields using your init_neumann function.
    U, V = init_neumann(n, center, i_value_U, i_value_V)

    # Create figure and axis.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Gray-Scott Model: U Concentration")
    plt.colorbar(ax.imshow(U[1:-1, 1:-1], cmap='jet', interpolation='nearest'), ax=ax)

    frames = []
    num_frames = 1000
    for frame in range(num_frames):
        U, V = update(U, V, dx, dt, dU, dV, kill=kill, feed=feed)
        # Eliminate ghost cells from image
        im = ax.imshow(U[1:-1, 1:-1], cmap='jet', interpolation='nearest', animated=True)
        frames.append([im])

    ani = animation.ArtistAnimation(fig, frames, interval=20, blit=True, repeat_delay=1000)
    ani.save('gray_scott.gif', writer='pillow', fps=50)    
    plt.show()


# Optional
def implement_mask():
    """Implement a mask
    - For which parameters?"""
    return

# n = 900
# center = 30

# dx = 1.0
# dt = 1.0

# dU = 0.16
# dV = 0.08

# feed = 0.035
# kill = 0.06

# U, V = init_neumann(n, center, i_value_U=0.5, i_value_V=0.25)

# t = 0
# while t < 1000:
#     U, V = U, V = update(U, V, dx, dt, dU, dV, kill=0.06, feed=0.035)
#     t += 1
