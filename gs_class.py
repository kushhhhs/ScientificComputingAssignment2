"""This module answers assignment 2.3 and will later likely be divided 
into seperate modules"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class GrayScott():
    """This class models A 2D Gray-Scott reaction-diffusion equation."""

    def __init__(
        self,
        n: int = 100,
        center: int = 6,
        i_value_U: float = 0.5,
        i_value_V: float = 0.25,
        dx: float = 1.0,
        dt: float = 1.0,
        dU: float = 0.16,
        dV: float = 0.08,
        feed: float = 0.035,
        kill: float = 0.06
    ):
        self.dx = dx
        self.dt = dt
        self.dU = dU
        self.dV = dV
        self.feed = feed
        self.kill = kill
        self.U, self.V = self.init_neumann(n, center, i_value_U, i_value_V)

    def init_neumann(self, n, center, i_value_U, i_value_V):
        """Initializes the GS model"""

        U = np.zeros((n, n))
        V = np.zeros((n, n))

        # Initializes all U values at i_value_U
        U[:,:] = i_value_U

        # Sets value of V to center positions
        V = self.center_positions(n, center, V, i_value_V)

        # Apply von Neuman
        U = self.boundary_neumann(U)
        V = self.boundary_neumann(V)

        return U, V

    def center_positions(self, n, c, V, i_value_V):
        """Gets the center positions for the grid square mask"""
        start = (n - c) // 2
        end = start + c
        
        for i in range(start, end):
            for j in range(start, end):
                V[i, j] = i_value_V

        return V

    def boundary_neumann(self, grid):
        """Implements the Neumann Boundary Conditions
        e.g. the derivative at the boundary is zero (no flux across boundary)"""
        f = np.pad(grid, 1)

        f[:, 0] = f[:, 1]
        f[:, -1] = f[:, -2]
        f[0, :] = f[1, :]
        f[-1, :] = f[-2, :]

        return f

    def laplacian_neumann(self, grid):
        """Computes Laplacian for von Neumann BC"""
        top = grid[:-2, 1:-1]
        bottom = grid[2:, 1:-1]
        left = grid[1:-1, :-2]
        right = grid[1:-1, 2:]
        
        return ((top + bottom + left + right - 4*grid[1:-1, 1:-1]) / self.dx**2)

    def gray_scott(self, LU, LV, U, V, noise_strength=0.0):
        """Calculates The gray scott reaction-diffusion equations.
        LU and LV: Laplacian of U and V
        dU and dV: respective diffusion coefficients
        standard noise strength set to zero so no noise present"""
        noise = noise_strength * np.random.normal(loc=0, scale=1.0, size=U.shape)
        
        uv2 = U * V**2

        dUdt = (self.dU * LU) - uv2 + self.feed*(1 - U) + noise
        dVdt = (self.dV * LV) + uv2 - V*(self.feed + self.kill) + noise

        return dUdt, dVdt

    def update(self):
        """Updates the grid for U and V (seperate grids)"""
        
        # Compute the Laplacian for U and V
        LU = self.laplacian_neumann(self.U)
        LV = self.laplacian_neumann(self.V)
        
        # Compute the Gray-Scott reaction
        dUdt, dVdt = self.gray_scott(LU, LV, self.U[1:-1, 1:-1], self.V[1:-1, 1:-1])

        # Update U and V
        self.U[1:-1, 1:-1] += self.dt * dUdt
        self.V[1:-1, 1:-1] += self.dt * dVdt

        # Apply the Neumann boundary conditions
        U_new = self.boundary_neumann(self.U[1:-1, 1:-1])
        V_new = self.boundary_neumann(self.V[1:-1, 1:-1])
        
        self.U = U_new
        self.V = V_new


def create_animation2(n, center, frames_per_update):
    """Creates an animation of U using FuncAnimation"""
    
    gs1 = GrayScott(n=n, center=center)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Gray-Scott Model: U Concentration")

    im = ax.imshow(gs1.U[1:-1, 1:-1], cmap='jet', interpolation='nearest')
    plt.colorbar(im, ax=ax)

    def update_frame(frame):
        """Updates U and the image data"""
        for f in range(10):
            gs1.update()
        im.set_array(gs1.U[1:-1, 1:-1])

        return [im]

    ani = animation.FuncAnimation(fig, update_frame, frames=1000, interval=20, blit=False, repeat_delay=1000)

    ani.save('gray_scott_func2.gif', writer='pillow', fps=50)    
    plt.show()


def plot_field(iterations=500):
    gs = GrayScott()

    for _ in range(iterations):
        gs.update()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    im1 = ax1.imshow(gs.U[1:-1, 1:-1], cmap="jet", origin="lower")
    ax1.set_title("Concentration Field of U")
    fig.colorbar(im1, ax=ax1, orientation="vertical", label="Concentration of U")
    im2 = ax2.imshow(gs.V[1:-1, 1:-1], cmap="jet", origin="lower")
    ax2.set_title("Concentration Field of V")
    fig.colorbar(im2, ax=ax2, orientation="vertical", label="Concentration of V")
    
    #plt.title(f"Gray-Scott Reaction-Diffusion at t={iterations} for U and V")
    plt.show()
    return

plot_field()


# create_animation2(n=100, center=6, frames_per_update=10)