import os
import numpy as np
from evoman.environment import Environment
from controller1 import player_controller
from evoman.controller import Controller

# Configuration
experiment_name = 'optimization_test'
os.makedirs(experiment_name, exist_ok=True)

n_hidden_neurons_1 = 10
n_hidden_neurons_2 = 5  
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initialize Environment
env = Environment(
    experiment_name=experiment_name,
    enemies=[5],
    playermode="ai",
    player_controller=player_controller(n_hidden_neurons_1, n_hidden_neurons_2),  # Pass two hidden layers
    enemymode="static",
    level=2,
    speed="fastest",
    visuals=False
)

# Genetic Algorithm Parameters
npopulation = 200       # Population size
gens = 30               # Number of generations
mutation_rate = 0.1    # Mutation rate
dom_u, dom_l = 2, -2    # Upper and lower bounds of the weights

# Number of variables in the controller
n_vars = (
    (env.get_num_sensors() + 1) * n_hidden_neurons_1 +  # Weights and biases from input -> hidden layer 1
    (n_hidden_neurons_1 + 1) * n_hidden_neurons_2 +  # Weights and biases from hidden layer 1 -> hidden layer 2
    (n_hidden_neurons_2 + 1) * 5  # Weights and biases from hidden layer 2 -> output layer (5 actions)
)

# Run the sim and return the fitness
def simulate(x):
    f, _, _, _ = env.play(pcont=x)
    return f

# Evaluate the current population
def evaluate(population):
    return np.array([simulate(individual) for individual in population])

# Tournament Selection
def tournament_selection(population, fitness, k=5):
    selected = []
    for _ in range(len(population)):
        contenders = np.random.choice(len(population), k, replace=False)
        winner = contenders[np.argmax(fitness[contenders])]
        selected.append(population[winner])
    return np.array(selected)

# Uniform Crossover (Could maybe change this)
def crossover(parent1, parent2):
    mask = np.random.rand(n_vars) < 0.5
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2

# Gaussian Mutation (Could maybe change this)
def mutate(child):
    for i in range(n_vars):
        if np.random.rand() < mutation_rate:
            child[i] += np.random.normal(0, 0.1)
            child[i] = np.clip(child[i], dom_l, dom_u)
    return child

# Initialize population
population = np.random.uniform(dom_l, dom_u, (npopulation, n_vars))
fitness = evaluate(population)

# Genetic Algorithm Loop
for generation in range(1, gens + 1):
    # Selection
    selected = tournament_selection(population, fitness)
    
    # Crossover
    offspring = []
    for i in range(0, npopulation, 2):
        parent1, parent2 = selected[i], selected[i+1]
        child1, child2 = crossover(parent1, parent2)
        # child1, child2 = two_point_crossover(parent1, parent2)
        offspring.extend([child1, child2])
    offspring = np.array(offspring)[:npopulation]
    
    # Mutation
    offspring = np.array([mutate(child) for child in offspring])
    
    # Evaluation
    offspring_fitness = evaluate(offspring)
    
    # Replacement: Elitism (keep the best individual) (Could maybe change this)
    best_idx = np.argmax(fitness)
    worst_idx = np.argmin(offspring_fitness)
    if fitness[best_idx] > offspring_fitness[worst_idx]:
        offspring[worst_idx] = population[best_idx]
        offspring_fitness[worst_idx] = fitness[best_idx]
    
    population, fitness = offspring, offspring_fitness
    
    # Logging
    best_fitness = np.max(fitness)
    mean_fitness = np.mean(fitness)
    print(f'Generation {generation}: Best Fitness = {best_fitness:.4f}, Mean Fitness = {mean_fitness:.4f}')

# Save Best Solution
best_idx = np.argmax(fitness)
best_solution = population[best_idx]
np.savetxt(os.path.join(experiment_name, 'best_solution.txt'), best_solution)
print(f'Best solution saved with fitness: {fitness[best_idx]:.4f}')