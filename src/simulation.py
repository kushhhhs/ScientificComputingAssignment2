"""
Diffusion-Limited Aggregation (DLA) Simulation

This module simulates a DLA process on a 2D grid using a stochastic 
growth model. It includes functions for grid initialization, diffusion 
calculation, and visualization using Matplotlib animations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from numba import jit
import time
from scipy.special import erfc


def analytical_solution(x, t, D, max_range=10):
    """
    Computes the analytical solution for the diffusion equation.
    
    Parameters:
    - x (float or ndarray): Position(s) in the spatial domain.
    - t (float): Time at which the solution is evaluated.
    - D (float): Diffusion coefficient
    - max_range (int, optional): Number of terms to sum.

    Returns an ndarray with the computed analytical concentration values.
    """
    sum_analytical = np.zeros_like(x)
    for i in range(max_range):
        sum_analytical += erfc((1 - x + 2 * i) / (2*np.sqrt(D*t))) - erfc((1 + x + 2 * i) / (2*np.sqrt(D*t))) #Analytical solution
    
    return sum_analytical


def create_initial_grid(N):
    """
    Initializes a 2D grid for simulation with boundary conditions and a 
    seed particle.

    Parameters:
    - N (int): Size of the square grid (NxN).

    Returns:
    - tuple: (grid, cluster_points)
        - grid (ndarray): NxN array initialized with boundary conditions.
        - cluster_points (list): List containing the initial seed 
          particle position.
    """
    grid = np.zeros((N, N))

    # Setting Inital Boundary conditions
    grid[-1, :] = 1.0 
    cluster_points = [(0, N//2)]

    # Initial Seed particle
    grid[0, N//2] = 0.0

    return grid, cluster_points


def get_initial_conc(N):
    """
    Generates the initial concentration grid using an analytical 
    solution.

    Parameters:
    - N (int): Size of the grid (NxN).

    Returns a 2D array representing the initial concentration field.
    """
    L = 1 # Length scale 
    T = 1 # Time at which analytical solution is to be determined
    D = 1 # Diffusion Coefficient 

    y_values = np.linspace(0, L, N)
    analytic_vals = [float(analytical_solution(y, T, D)) for y in y_values]
    grid = np.array([[val]*N for val in analytic_vals])

    return grid


def cluster_to_array(cluster_points, N):
    """Converts a list of cluster points into a 2D boolean array."""
    cluster_array = np.zeros((N, N), dtype=np.bool_)

    for i, j in cluster_points:
        cluster_array[i, j] = True

    return cluster_array


def find_neighbours(cluster_points, N):
    """
    Identifies the neighboring grid points available for cluster growth.

    Parameters:
    - cluster_points (list of tuples): List of current cluster 
      coordinates.
    - N (int): Size of the grid.

    Returns a list of available neighboring points for growth.
    """
    cluster_set = set(cluster_points)
    neighbours = set()
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for i, j in cluster_points:
        for di, dj in directions:
            ni, nj = i + di, j + dj

            if (0 <= ni < N and 0 <= nj < N and (ni, nj) not in cluster_set):
                neighbours.add((ni, nj))

    return list(neighbours)


def calculate_growth_probability(grid, neighbours, eta=1.0):
    """
    Computes the probability distribution for growth based on 
    concentration.

    Parameters:
    - grid (ndarray): Current concentration field.
    - neighbours (list of tuples): List of available neighboring points.
    - eta (float, optional): Growth exponent controlling bias 
      (default=1.0).

    Returns a dictionary mapping each neighbor to its normalized growth 
    probability.
    """
    probabilities = {}
    normalized_probabilities = {}
    total_probability = 0.0
    
    for i, j in neighbours:
        prob = grid[i, j] ** eta  
        probabilities[(i, j)] = prob
        total_probability += prob
        
    if total_probability > 0:
        for point, prob in probabilities.items():
            normalized_probabilities[point] = prob / total_probability
            
    return normalized_probabilities


def select_growth_point(normalized_probabilities):
    """
    Selects a new cluster growth point based on the computed 
    probabilities.
    """
    if normalized_probabilities:
        points = list(normalized_probabilities.keys())
        probs = list(normalized_probabilities.values())
        selected_point = random.choices(points, weights=probs, k=1)[0]

        return selected_point

    else:
        print('No new point has been selected')

        return None


@jit(nopython=True)
def get_next_grid(grid, N, omega=1.7, cluster_array=None):
    """
    Performs a Successive Over-Relaxation (SOR) update on the grid.

    Parameters:
    - grid (ndarray): The current concentration grid.
    - N (int): Size of the grid.
    - omega (float, optional): Relaxation parameter for SOR. Default is 1.7.
    - cluster_array (ndarray, optional): Boolean array marking cluster 
      locations.
      If None, a zero array is created.

    Returns:
    - tuple: (new_grid, max_diff)
        - new_grid (ndarray): Updated concentration grid.
        - max_diff (float): Maximum absolute change in concentration.
    """
    new_grid = np.copy(grid)
    max_diff = 0.0
    
    if cluster_array is None:
        cluster_array = np.zeros((N, N), dtype=np.bool_)
        
    for i in range(1, N - 1):
        for j in range(N):
             # Does not update values if point belongs to a cluster
            if not cluster_array[i, j]: 
                old_value = grid[i, j]
                new_value = (0.25 * (new_grid[i-1, j] + grid[i+1, j] 
                    + new_grid[i, (j-1)%N] + grid[i, (j+1)%N]))
                new_grid[i, j] = (1 - omega) * old_value + omega * new_value
                max_diff = max(max_diff, abs(new_grid[i, j] - old_value))

    return new_grid, max_diff


def solve_laplace(grid, cluster_points, omega=1.7, max_iterations=10000, tol=1e-5):
    """Solves the time independent diffusion equation"""
    N = grid.shape[0]
    cluster_array = cluster_to_array(cluster_points, N)
    c = np.copy(grid)
    
    for iters in range(max_iterations):
        c, max_diff = get_next_grid(c, N, omega, cluster_array)

        if max_diff < tol:
            break

        if np.max(c) > 10 * np.max(grid):
            return None, iters
            
    return c, iters


def dla_simulation(N, num_particles=None, omega=1.7, eta=1.0):
    """
    Runs the Diffusion-Limited Aggregation (DLA) simulation.

    Parameters:
    - N (int): Size of the grid (NxN).
    - num_particles (int, optional): Number of particles to add.
    - omega (float, optional): Relaxation parameter for Laplace solver 
      (default=1.7).
    - eta (float, optional): Exponent for growth probability 
      (default=1.0).

    Returns:
    - tuple: (frames, final_concentration, avg_iterations)
        - frames (list): List of cluster states over time.
        - final_concentration (ndarray): Final concentration field.
        - avg_iterations (float): Average number of iterations per step.
    """
    # Create initial grid and cluster
    grid, cluster_points = create_initial_grid(N)
    iterations = []
    frames = [cluster_points.copy()]
    particles = 1
    
    # Continue until we reach num_particles or hit the top boundary
    while num_particles is None or particles < num_particles:
        conc, iters = solve_laplace(grid, cluster_points, omega)
        iterations.append(iters)
        
        if conc is None:
            print("Simulation failed: numerical instability detected")
            return None, None, None
            
        neighbours = find_neighbours(cluster_points, N)
        normalized_probs = calculate_growth_probability(conc, neighbours, eta)
        new_point = select_growth_point(normalized_probs)
        
        if new_point:
            i, j = new_point
            cluster_points.append(new_point)
            grid[new_point] = 0.0
            particles += 1
            frames.append(cluster_points.copy())
            
            # Check for reaching top boundary 
            if i == N - 1:
                print(f"Simulation stopped: cluster reached the top boundary after {particles} particles.")
                break
        
        else:
            print(f'Numerical scheme unstable for omega: {omega}. Please change settings.')
            break
            
    return frames, conc, np.mean(iterations)


def optimized_dla_simulation(N, num_particles, omega=1.7, eta=2.0):
    """
    Runs an optimized DLA simulation with an analytical initial 
    condition.

    Parameters:
    - N (int): Grid size (NxN).
    - num_particles (int): Number of particles to add.
    - omega (float, optional): Relaxation parameter for the SOR method. 
      Default is 1.7.
    - eta (float, optional): Exponent for growth probability calculation. 
      Default is 2.0.

    Returns:
    - tuple: (frames, final_concentration, average_iterations)
        - frames (list): List of cluster states over time.
        - final_concentration (ndarray): Final concentration grid.
        - average_iterations (float): Mean iterations per step.
    """
    # Create initial grid and cluster using analytical solution
    conc = get_initial_conc(N)
    cluster_points = [(0, N//2)]
    conc[0, N//2] = 0.0
    
    frames = [cluster_points.copy()]
    iterations = []
    particles = 1
    
    while particles < num_particles:
        conc, iters = solve_laplace(conc, cluster_points, omega)
        iterations.append(iters)
        
        if conc is None:
            print("Simulation failed: numerical instability detected")
            return None, None, None
            
        neighbours = find_neighbours(cluster_points, N)
        normalized_probs = calculate_growth_probability(conc, neighbours, eta)
        new_point = select_growth_point(normalized_probs)
        
        if new_point:
            cluster_points.append(new_point)
            conc[new_point] = 0.0
            particles += 1
            frames.append(cluster_points.copy())

        else:
            print(f'Numerical scheme unstable for omega: {omega}. '
                'Please change settings.')
            break
            
    return frames, conc, np.mean(iterations)


def animate_dla(frames, N, interval=100):
    """
    Generates an animation of DLA growth.

    Parameters:
    - frames (list): List of cluster states over time.
    - N (int): Size of the grid.
    - interval (int, optional): Time interval between frames 
      (default=100 ms).

    Returns a Matplotlib animation object.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, N)
    ax.set_ylim(-0.5, N)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("DLA Simulation (Cluster Points in Red)")

    scatter = ax.scatter([], [], color='red', s=0.5)

    def update(frame):
        cluster_points = frame
        if cluster_points:
            y_vals, x_vals = zip(*cluster_points)  # Swapping to (x, y)
        else:
            x_vals, y_vals = [], []
        scatter.set_offsets(np.c_[x_vals, y_vals])  
        return scatter,

    ani = FuncAnimation(fig, update, frames=frames, interval=interval, 
        blit=True)
    plt.close(fig)

    return ani


def plot_final_dla_with_concentration(frames, conc, eta):
    """
    Plots the final DLA cluster overlaid with a concentration heatmap.
    
    Parameters:
    - frames (list of lists): List of cluster states throughout the 
      simulation.
    - conc (ndarray): Final concentration grid.
    - eta (float): Growth probability exponent.

    Displays:
    - A heatmap representing the final concentration.
    - Cluster points in red.
    """
    N = len(conc)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"DLA Cluster with Heatmap (Eta = {eta})",fontweight='bold', 
        fontsize=16)

    heatmap = ax.imshow(conc, origin='upper', cmap='viridis', 
        interpolation='none')

    # Plot cluster points in red
    final_frame = frames[-1]
    if final_frame:
        y_vals, x_vals = zip(*final_frame)  
        ax.scatter(x_vals, y_vals, color='red', s=1.0)

    cbar = plt.colorbar(heatmap, ax=ax, label='Concentration')
    cbar.ax.tick_params(labelsize=16)  # Increase font size of colorbar ticks
    cbar.set_label('Concentration', fontsize=18)    
    plt.tight_layout()
    plt.show()


def check_optimal_sor(N=100, num_particles=50, runs=100):
    """
    Determines the optimal SOR relaxation parameter by testing various 
    values.

    Parameters:
    - N (int, optional): Grid size (NxN). Default is 100.
    - num_particles (int, optional): Number of particles to simulate. 
      Default is 50.
    - runs (int, optional): Number of runs per omega value. 
      Default is 100.

    Returns ether an optimal omega value if found, otherwise None.

    Displays a plot showing average iterations vs. omega.
    """
    omega_list = np.linspace(1.5, 1.95, num=50)
    avg_iterations_dict = {}

    for omega in omega_list:
        print(f'Testing omega = {omega}')
        iterations = []
        
        # Run multiple tests with this omega
        for _ in range(runs):
            _, _, avg_iters = optimized_dla_simulation(N, 
                num_particles=num_particles, omega=omega)
            if avg_iters is not None:
                iterations.append(avg_iters)
        
        if iterations:
            avg_iterations_dict[omega] = sum(iterations) / len(iterations)
        else:
            print(f"Omega = {omega} caused numerical instability")
    

    plt.figure(figsize=(8, 5))
    plt.plot(list(avg_iterations_dict.keys()), 
        list(avg_iterations_dict.values()), marker='o', linestyle='-')
    plt.xlabel("Omega", fontsize = 18)
    plt.ylabel("Average Iterations", fontsize = 18)
    plt.title("Average Iterations vs Omega Parameter", fontweight='bold', 
        fontsize = 20)
    plt.grid(True)
    plt.savefig("omega_optimization.pdf")
    plt.show()
    

    if avg_iterations_dict:
        optimal_omega = min(avg_iterations_dict, key=avg_iterations_dict.get)
        print(f"Optimal omega value: {optimal_omega}")
        return optimal_omega

    else:
        print("Could not determine optimal omega value")
        return None