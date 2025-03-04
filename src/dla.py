import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from numba import njit

def create_initial_grid(N):
    '''
    Create an initial grid for simulation
    '''
    grid = np.zeros((N, N))
    grid[-1, :] = 1.0 
    cluster_points = [(0,N//2)]
    grid[ 0, N//2] = 1.0  # Seed particle

    return grid, cluster_points

def cluster_to_array(cluster_points, N):
    '''
    Converts the list of cluster points into a 2D array of size NXN initialized with False 
    If the point is part of cluster changes the value of the boolean array to True.
    Helps check if a point is in the cluster instead of searching through the list of cluster points everytime
    '''
    cluster_array = np.zeros((N, N), dtype=np.bool_)
    for i, j in cluster_points:
        cluster_array[i, j] = True
    return cluster_array

def find_neighbours(cluster_points, N):
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
    probabilities = {}
    normalized_probabilities={}
    total_probability = 0.0
    for i, j in neighbours:
        prob = grid[i, j] ** eta  
        probabilities[(i, j)] = prob
        total_probability += prob
    if total_probability > 0:
        normalized_probabilities={}
        for point, prob in probabilities.items():
            normalized_probabilities[point] = prob/ total_probability
    return normalized_probabilities

def select_growth_point(neighbours, normalized_probabilities):
    if normalized_probabilities:
        points = list(normalized_probabilities.keys())
        probs = list(normalized_probabilities.values())
        selected_point = random.choices(points, weights=probs, k=1)[0]
        return selected_point
    else:
        print('No new point has been selected')
        return None 

@njit
def get_next_grid(grid, N, omega=1.7, cluster_array=None):
    new_grid = np.copy(grid)
    max_diff = 0.0
    for i in range(1, N - 1):
        for j in range(N):
            if not cluster_array[i, j]: # Doesnot update the values if the point belongs to a cluster
                old_value = grid[i, j]
                new_value = 0.25 * (new_grid[i-1, j] + grid[i+1, j] + new_grid[i, (j-1)%N] + grid[i, (j+1)%N])
                new_grid[i, j] = (1 - omega) * old_value + omega * new_value
                max_diff = max(max_diff, abs(new_grid[i, j] - old_value))
    return new_grid, max_diff

def solve_laplace(grid, cluster_points, omega=1.7, max_iterations=1000, tol=1e-5):
    '''
    Solves the time independent diffusion equation and returns the steady state concentration grid
    '''
    N = grid.shape[0]
    cluster_array = cluster_to_array(cluster_points, N)# Converts all the cluster points into an N X N boolean array with either true of false to indicate whether a point is in the cluster
    c = np.copy(grid)
    for _ in range(max_iterations):
        c, max_diff = get_next_grid(c, N, omega, cluster_array)
        if max_diff < tol:
            break
    return c # Returns the steady state concentration grid 

def dla_simulation(N, num_particles, omega=1.7, eta=1.0):
    '''
    Function to Simulate DLA growth model 
    '''
    # Create an intial grid with Top Row having concentration values of 1.0 and set the initial seed to a point at the bottom of the domain
    grid, cluster_points = create_initial_grid(N)

    frames = []  
    points =1
    for _ in range(num_particles):
  
        conc = solve_laplace(grid, cluster_points, omega)# Solve the Laplace Equation to find the concentration values at steady state

        neighbours = find_neighbours(cluster_points, N)# Find the neighbours of the cluster 
        normalized_probs = calculate_growth_probability(conc, neighbours, eta)
        new_point = select_growth_point(neighbours, normalized_probs)
        if new_point:
            points +=1
            cluster_points.append(new_point)
            grid[new_point] = 1.0  # Set the new cluster point
        else:
            print(f'{points} Maximum cluster point')
            return frames, conc
        frames.append(cluster_points.copy())  # Store current state

    return frames, conc

# Animation Function
def animate_dla(frames, N, interval=100):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, N)
    ax.set_ylim(-0.5, N)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("DLA Simulation (Cluster Points in Red)")

    scatter = ax.scatter([], [], color='red', s=2)

    def update(frame):
        cluster_points = frame
        if cluster_points:
            y_vals, x_vals = zip(*cluster_points)  # Swapping to (x, y)
        else:
            x_vals, y_vals = [], []
        scatter.set_offsets(np.c_[x_vals, y_vals])  
        return scatter,

    ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
    plt.show()

# Run DLA Simulation and Animation
N = 100
num_particles = 300

frames, conc = dla_simulation(N, num_particles)
animate_dla(frames, N)
# plt.figure(figsize=(8, 8))
# plt.imshow(conc, cmap='hot', interpolation='nearest') 
# plt.colorbar(label='Concentration')  
# plt.title("Concentration Gradient")

#
# plt.xlabel("X-axis (Horizontal position)")
# plt.ylabel("Y-axis (Vertical position)")
# plt.gca().invert_yaxis()

# plt.show()
# exit()