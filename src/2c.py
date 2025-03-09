"""This module contains the first functions to implement 2.d. It is no 
longer neccesary to run the code as functions have been incorporated 
into the GrayScott class. Some periodic boundary conditions are here. 
Those have not been implemented."""

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


def gray_scott_noise(dU, dV, LU, LV, U, V, feed, kill, noise_strength):
    """Calculates The gray scott reaction-diffusion equations.
    LU and LV: Laplacian of U and V
    dU and dV: respective diffusion coefficients"""
    noise = noise_strength * np.random.normal(loc=0, scale=1.0, size=U.shape)

    # Some stuff to calculate noise
    dUdt = (dU * LU) - (U * V**2) + feed*(1 - U) + noise
    dVdt = (dV * LV) + (U * V**2) - V*(feed + kill) + noise

    return dUdt, dVdt


def update(U, V, dx, dt, dU, dV, kill=0.06, feed=0.035):
    """Updates the grid for U and V (seperate grids)"""
    
    # Compute the Laplacian for U and V
    LU = laplacian_neumann(U, dx)
    LV = laplacian_neumann(V, dx)
    
    # Compute the Gray-Scott reaction
    dUdt, dVdt = gray_scott_noise(dU, dV, LU, LV, U[1:-1, 1:-1], V[1:-1, 1:-1], feed, kill, noise_strength=0.001)

    # Update U and V
    U[1:-1, 1:-1] += dt * dUdt
    V[1:-1, 1:-1] += dt * dVdt

    # Apply the Neumann boundary conditions
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

    # Initializes all U values at i_value_U
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



def create_animation(n, center, frames_per_update, dx=1.0, dt=1.0, dU=0.16, dV=0.08, feed=0.035, kill=0.06, i_value_U=0.5, i_value_V=0.25):
    """Creates an animation of U using FuncAnimation"""
    
    U, V = init_neumann(n, center, i_value_U, i_value_V)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Gray-Scott Model: U Concentration")
    
    im = ax.imshow(U[1:-1, 1:-1], cmap='jet', interpolation='nearest')
    plt.colorbar(im, ax=ax)

    def update_frame(frame, frames_per_update):
        """Updates U and the image data"""
        nonlocal U, V
        for f in range(frames_per_update):
            U, V = update(U, V, dx, dt, dU, dV, kill=kill, feed=feed)
        im.set_array(U[1:-1, 1:-1])

        return [im]

    ani = animation.FuncAnimation(fig, update_frame, frames=100, interval=20, blit=False, repeat_delay=1000, fargs=(frames_per_update,))

    ani.save('gray_scott_func.gif', writer='pillow', fps=50)    
    plt.show()



    