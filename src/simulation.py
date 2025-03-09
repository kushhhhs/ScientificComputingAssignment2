import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from numba import jit
import time
from scipy.special import erfc


def analytical_solution(x, t, D, max_range=10):
    """
    Calculate analytical solution for diffusion.
    
    Parameters:
    -----------
    D: diffusion coefficient
    x: space/place
    t: time at which to evaluate
    max_range: range for summation
    """
    sum_analytical = np.zeros_like(x)
    for i in range(max_range):
        sum_analytical += erfc((1 - x + 2 * i) / (2*np.sqrt(D*t))) - erfc((1 + x + 2 * i) / (2*np.sqrt(D*t))) #Analytical solution
    return sum_analytical

def create_initial_grid(N):
    """Create an initial grid for simulation with a seed particle"""
    grid = np.zeros((N, N))
    grid[-1, :] = 1.0 # Setting Inital Boundary conditions
    cluster_points = [(0, N//2)]
    grid[0, N//2] = 0.0  #  Initial Seed particle
    return grid, cluster_points

def get_initial_conc(N):
    """Create initial concentration using analytical solution"""
    L = 1 #Length scale of the simulation 
    T = 1 #Time at which analytical solution is to be determined
    D = 1 #Diffusion Coefficient 
    y_values = np.linspace(0, L, N)
    analytic_vals = [float(analytical_solution(y, T, D)) for y in y_values]
    grid = np.array([[val]*N for val in analytic_vals])
    return grid

def cluster_to_array(cluster_points, N):
    """Converts the list of cluster points into a 2D boolean array"""
    cluster_array = np.zeros((N, N), dtype=np.bool_)
    for i, j in cluster_points:
        cluster_array[i, j] = True
    return cluster_array

def find_neighbours(cluster_points, N):
    """Finds all the nearest probable neighbours"""
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
    """Calculates the probability for the cluster to grow to each neighboring point"""
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
    """Select a growth point based on normalized probabilities"""
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
    """Numba-optimized function to update the grid using SOR method"""
    new_grid = np.copy(grid)
    max_diff = 0.0
    
    if cluster_array is None:
        cluster_array = np.zeros((N, N), dtype=np.bool_)
        
    for i in range(1, N - 1):
        for j in range(N):
            if not cluster_array[i, j]:  # Does not update values if point belongs to a cluster
                old_value = grid[i, j]
                new_value = 0.25 * (new_grid[i-1, j] + grid[i+1, j] + new_grid[i, (j-1)%N] + grid[i, (j+1)%N])
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
    Run the standard DLA simulation
    
    Parameters:
    -----------
    N : int
        Grid size (N x N)
    num_particles : int or None
        Number of particles to add, or None to run until top boundary is reached
    omega : float
        Relaxation parameter for SOR method
    eta : float
        Exponent for growth probability calculation
    
    Returns:
    --------
    tuple: (frames, final_concentration, average_iterations)
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
    Run the optimized DLA simulation with analytical initial conditions
    
    Parameters:
    -----------
    N : int
        Grid size (N x N)
    num_particles : int
        Number of particles to add
    omega : float
        Relaxation parameter for SOR method
    eta : float
        Exponent for growth probability calculation
    
    Returns:
    --------
    tuple: (frames, final_concentration, average_iterations)
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
            print(f'Numerical scheme unstable for omega: {omega}. Please change settings.')
            break
            
    return frames, conc, np.mean(iterations)

# Visualization functions
def animate_dla(frames, N, interval=100):
    """Create animation of DLA growth"""
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

    ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
    plt.close(fig)
    #plt.show()

    return ani

def plot_final_dla_with_concentration(frames, conc, eta):
    """Plot the final DLA cluster with concentration heatmap"""
    N = len(conc)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"DLA Cluster with Heatmap (Eta = {eta})",fontweight='bold', fontsize=16)

    heatmap = ax.imshow(conc, origin='upper', cmap='viridis', interpolation='none')

    # Plot cluster points in red
    final_frame = frames[-1]
    if final_frame:
        y_vals, x_vals = zip(*final_frame)  
        ax.scatter(x_vals, y_vals, color='red', s=1.0)

    plt.colorbar(heatmap, ax=ax, label='Concentration')
    plt.tight_layout()
    plt.show()

def check_optimal_sor(N=100, num_particles=50, runs=100):
    """Find optimal SOR parameter by testing different values"""
    omega_list = np.linspace(1.5, 1.95, num=50)
    avg_iterations_dict = {}

    for omega in omega_list:
        print(f'Testing omega = {omega}')
        iterations = []
        
        # Run multiple tests with this omega
        for _ in range(runs):
            _, _, avg_iters = optimized_dla_simulation(N, num_particles=num_particles, omega=omega)
            if avg_iters is not None:
                iterations.append(avg_iters)
        
        if iterations:
            avg_iterations_dict[omega] = sum(iterations) / len(iterations)
        else:
            print(f"Omega = {omega} caused numerical instability")
    

    plt.figure(figsize=(8, 5))
    plt.plot(list(avg_iterations_dict.keys()), list(avg_iterations_dict.values()), marker='o', linestyle='-')
    plt.xlabel("Omega", fontsize = 18)
    plt.ylabel("Average Iterations", fontsize = 18)
    plt.title("Average Iterations vs Omega Parameter", fontweight='bold', fontsize = 20)
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