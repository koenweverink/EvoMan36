import os
import sys
import numpy as np
from evoman.environment import Environment
from controller1 import player_controller

# Config
experiment_name = 'optimization_test'
best_solution_file = os.path.join(experiment_name, 'best_solution.txt')

# Check if the best solution file exists
if not os.path.exists(best_solution_file):
    print(f"Best solution file '{best_solution_file}' not found.")
    sys.exit(1)

# Load the best solution
best_solution = np.loadtxt(best_solution_file)

# Number of hidden neurons (needs to match environment.py)
n_hidden_neurons = 10

# Initialize Environment with Visuals Enabled
env = Environment(
    experiment_name=experiment_name,
    enemies=[2],  # Ensure this matches your training setup
    playermode="ai",
    player_controller=player_controller(n_hidden_neurons),  # Adjust if different
    enemymode="static",
    level=2,
    speed="normal",  # You can set to 'slow', 'normal', or 'fastest'
    visuals=True      # Enable visuals
)

# Play the game with the best controller by passing pcont directly
f, p, e, t = env.play(pcont=best_solution)

# Print the results
print(f"Fitness: {f}")
print(f"Player Life: {p}")
print(f"Enemy Life: {e}")
print(f"Time: {t}")
