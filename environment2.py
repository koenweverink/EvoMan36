import os
import numpy as np
from evoman.environment import Environment
from controller2 import player_controller

# Configuration
experiment_name = 'environment2'
os.makedirs(experiment_name, exist_ok=True)

n_hidden_neurons = 10
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initialize Environment
env = Environment(
    experiment_name=experiment_name,
    enemies=[2],
    playermode="ai",
    player_controller=player_controller(n_hidden_neurons),
    enemymode="static",
    level=2,
    speed="fastest",
    visuals=False
)

# Genetic Algorithm Parameters
npopulation = 51       # population size, adjusted to be a multiple of three for crossover
gens = 30               # Number of generations
mutation_rate = 0.2     # Mutation rate
dom_u, dom_l = 1, -1    # Upper and lower bounds of the weights

# Number of variables in the controller
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

# Run the sim and return the fitness
def simulate(x):
    f, _, _, _ = env.play(pcont=x)
    return f

# Evaluate the current population
def evaluate(population):
    return np.array([simulate(individual) for individual in population])


# Normalize fitness
def normalize(x, fitness_pop):
    if ( max(fitness_pop) - min(fitness_pop) ) > 0:
        normalized_fitness = ( x - min(fitness_pop) )/( max(fitness_pop) - min(fitness_pop) )
    else:
        normalized_fitness = 0

    return normalized_fitness

# Tournament Selection for three parents
def tournament_selection(population, fitness, k=3):
    selected = []
    for _ in range(len(population) // k * k):  # ensures that only multiples of three are selected
        contenders = np.random.choice(len(population), k, replace=False)
        winner = contenders[np.argmax(fitness[contenders])]
        selected.append(population[winner])
    return np.array(selected)

# Three-parent crossover
def three_parent_crossover(parent1, parent2, parent3):
    n_genes = len(parent1)
    crossover_points = np.sort(np.random.choice(range(1, n_genes), 2, replace=False))
    child1 = np.empty(n_genes, dtype=parent1.dtype)
    child2 = np.empty(n_genes, dtype=parent1.dtype)
    child3 = np.empty(n_genes, dtype=parent1.dtype)

    child1[:crossover_points[0]] = parent1[:crossover_points[0]]
    child1[crossover_points[0]:crossover_points[1]] = parent2[crossover_points[0]:crossover_points[1]]
    child1[crossover_points[1]:] = parent3[crossover_points[1]:]

    child2[:crossover_points[0]] = parent2[:crossover_points[0]]
    child2[crossover_points[0]:crossover_points[1]] = parent3[crossover_points[0]:crossover_points[1]]
    child2[crossover_points[1]:] = parent1[crossover_points[1]:]

    child3[:crossover_points[0]] = parent3[:crossover_points[0]]
    child3[crossover_points[0]:crossover_points[1]] = parent1[crossover_points[0]:crossover_points[1]]
    child3[crossover_points[1]:] = parent2[crossover_points[1]:]

    return child1, child2, child3

# Mutation Function
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

    fitness_norm =  np.array(list(map(lambda y: normalize(y,fitness), fit_pop)))
    selected = tournament_selection(population, fitness)

    offspring = []

    for i in range(0, len(selected), 3):
        parent1, parent2, parent3 = selected[i], selected[i+1], selected[i+2]
        child1, child2, child3 = three_parent_crossover(parent1, parent2, parent3)
        offspring.extend([child1, child2, child3])

    offspring = np.array(offspring)[:npopulation]  # ensure population size remains constant
    offspring = np.array([mutate(child) for child in offspring])

    offspring_fitness = evaluate(offspring)

    # Replacement with Elitism: Keep the best individual from the current generation
    best_idx = np.argmax(fitness)
    worst_idx = np.argmin(offspring_fitness)
    if fitness[best_idx] > offspring_fitness[worst_idx]:
        offspring[worst_idx] = population[best_idx]
        offspring_fitness[worst_idx] = fitness[best_idx]

    population, fitness = offspring, offspring_fitness

    best_fitness = np.max(fitness)
    mean_fitness = np.mean(fitness)
    std_fitness = np.std(fitness)

    print(f'Generation {generation}: Best Fitness = {best_fitness:.4f}, Mean Fitness = {mean_fitness:.4f}, Std Fitness = {std_fitness:.4f}')

# Save Best Solution
best_idx = np.argmax(fitness)
best_solution = population[best_idx]
np.savetxt(os.path.join(experiment_name, 'best_solution.txt'), best_solution)
print(f'Best solution saved with fitness: {fitness[best_idx]:.4f}')
