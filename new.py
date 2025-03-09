import sys
sys.path.append("src")  
import simulation
N =100
eta = 2.0
frames, conc, _ = simulation.dla_simulation(N, num_particles = 100, eta= eta)
simulation.plot_final_dla_with_concentration(frames, conc, eta = eta)
ani = simulation.animate_dla(frames, N)