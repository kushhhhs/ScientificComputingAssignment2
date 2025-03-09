"""This module creates a class for the Gray Scott reaction-diffusion 
model."""

import numpy as np


class GrayScott():
    """This class creates A 2D Gray-Scott reaction-diffusion model."""

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
        kill: float = 0.06, 
        noise: float = 0.0
    ):
        """Initialises the Gray Scott class. and creates self.U and 
        self.V from initial neumann conditions.
        
        Parameters:
        - n (int): Size of the grid (nxn).
        - center (int): Size of the initial perturbation for V at the 
            center.
        - i_value_U (float): Initial concentration of U.
        - i_value_V (float): Initial concentration of V in the 
            perturbation.
        - dx (float): Grid spacing (spatial resolution).
        - dt (float): Time step for the numerical solver.
        - dU (float): Diffusion coefficient for U.
        - dV (float): Diffusion coefficient for V.
        - feed (float): Feed rate, controls the supply of U.
        - kill (float): Kill rate, controls the removal of V.
        - noise (float): Strength of the noise added to the system.
        """
        self.dx = dx
        self.dt = dt
        self.dU = dU
        self.dV = dV
        self.feed = feed
        self.kill = kill
        self.noise_strength = noise
        self.U, self.V = self.init_neumann(n, center, i_value_U, i_value_V)

    def init_neumann(self, n, center, i_value_U, i_value_V):
        """Initializes the GS model with passed parameters and calls 
        Neumann boundary conditions.
        
        Parameters:
        - n (int): Size of the grid (nxn).
        - center (int): Size of the size for V at the center.
        - i_value_U (float): Value for which U is set in the grid.
        - i_Value_V (float): Value for which V is set in the grid.
        
        Returns grids U and V which are subsequently set as self.U and 
        self.V.
        """

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
        """Gets the center positions for the grid center square mask and
        assigns it the correct value for V and returns the updated V 
        grid.
        
        Parameters:
        - n (int): Size of the grid.
        - c (int): Size of the central perturbation square.
        - V (np.array): Grid for the V concentration.
        - i_value_V (float): Initial value for V inside the perturbation.
        """

        # Start and end spots for center positions
        start = (n - c) // 2
        end = start + c
        
        for i in range(start, end):
            for j in range(start, end):
                V[i, j] = i_value_V

        return V

    def boundary_neumann(self, grid):
        """Implements the Neumann Boundary Conditions
        e.g. the derivative at the boundary is zero (no flux across 
        boundary)."""

        # Copy the given grid and place zeros around it
        f = np.pad(grid, 1)

        f[:, 0] = f[:, 1]
        f[:, -1] = f[:, -2]
        f[0, :] = f[1, :]
        f[-1, :] = f[-2, :]

        return f

    def laplacian_neumann(self, grid):
        """Computes Laplacian for von Neumann Boundary Conditions."""

        top = grid[:-2, 1:-1]
        bottom = grid[2:, 1:-1]
        left = grid[1:-1, :-2]
        right = grid[1:-1, 2:]
        
        return ((top + bottom + left + right - 4*grid[1:-1, 1:-1]) 
                 / self.dx**2)

    def gray_scott(self, LU, LV, U, V):
        """Calculates The gray scott reaction-diffusion equations.

        Parameters: 
        - LU (np.array): Laplacian of U.
        - LV (np.array): Laplacian of V.
        - U (np.array): Current state of U.
        - V (np.array): Current state of V.

        Returns:
        - dUdt (np.array): Rate of change of U.
        - dVdt (np.array): Rate of change of V.
        """

        noise = self.noise_strength * np.random.normal(loc=0, scale=1.0, 
            size=U.shape)
        
        uv2 = U * V**2

        dUdt = (self.dU * LU) - uv2 + self.feed*(1 - U) + noise
        dVdt = (self.dV * LV) + uv2 - V*(self.feed + self.kill) + noise

        return dUdt, dVdt

    def update(self):
        """Updates the grids for U and V according to Von Neumann
        boundary conditions. Sets the newly computed U and V as self.
        
        This method:
        1. Computes the Laplacian of U and V.
        2. Applies the Gray-Scott reaction-diffusion equations.
        3. Updates the U and V grids.
        4. Reapplies Neumann boundary conditions.
        """
        
        # Compute the Laplacian for U and V
        LU = self.laplacian_neumann(self.U)
        LV = self.laplacian_neumann(self.V)
        
        # Compute the Gray-Scott reaction
        dUdt, dVdt = self.gray_scott(LU, LV, self.U[1:-1, 1:-1], 
            self.V[1:-1, 1:-1])

        # Update U and V
        self.U[1:-1, 1:-1] += self.dt * dUdt
        self.V[1:-1, 1:-1] += self.dt * dVdt

        # Apply the Neumann boundary conditions
        U_new = self.boundary_neumann(self.U[1:-1, 1:-1])
        V_new = self.boundary_neumann(self.V[1:-1, 1:-1])
        
        self.U = U_new
        self.V = V_new