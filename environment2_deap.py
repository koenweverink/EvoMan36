import os
import numpy as np
from evoman.environment import Environment
from controller2 import player_controller

# Configuration
experiment_name = 'optimization_test'
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
npopulation = 50       # population size
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
def normalize(fitness, global_min, global_max):
    normalized_fitness = 100 * (fitness - global_min) / (global_max - global_min)
    return normalized_fitness

# Tournament Selection
def tournament_selection(population, fitness, k=2):
    selected = []
    for _ in range(len(population)):
        contenders = np.random.choice(len(population), k, replace=False)
        winner = contenders[np.argmax(fitness[contenders])]
        selected.append(population[winner])
    return np.array(selected)

# Shuffle Crossover
def crossover(parent1, parent2):
    # Generate a random shuffle order
    shuffle_indices = np.random.permutation(len(parent1))

    # Shuffle both parents
    shuffled_parent1 = parent1[shuffle_indices]
    shuffled_parent2 = parent2[shuffle_indices]

    # Perform single-point crossover on shuffled parents
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate([shuffled_parent1[:crossover_point], shuffled_parent2[crossover_point:]])
    child2 = np.concatenate([shuffled_parent2[:crossover_point], shuffled_parent1[crossover_point:]])
    
    # Restore the original order
    reverse_indices = np.argsort(shuffle_indices)
    child1 = child1[reverse_indices]
    child2 = child2[reverse_indices]

    return child1, child2

# Mutation with Swap positions
def mutate2(child):
    '''This applies a mutation operator to a list and returns the mutated list.'''

    for i in range(n_vars):
        if np.random.rand() < mutation_rate:
            child[i] += np.random.normal(0, 0.1)
            child[i] = np.clip(child[i], dom_l, dom_u)

    if np.random.uniform() < mutation_rate:
        i, j = np.random.choice(range(1, len(child) - 1), size=2, replace=False)
        child[i], child[j] = child[j], child[i]
        
    return child

# Doomsday function as replacement
def doomsday(pop, fit_pop):
    worst = int(npopulation / 10)  # a quarter of the population
    order = np.argsort(fit_pop)
    orderasc = order[:worst]
    best_dna = pop[order[-1]]

    for o in orderasc:
        for j in range(n_vars):
            if np.random.uniform(0, 1) <= np.random.uniform(0, 1):
                pop[o][j] = np.random.uniform(dom_l, dom_u)
            else:
                pop[o][j] = best_dna[j]
        fit_pop[o] = simulate(pop[o])
    return pop, fit_pop

# Initialize population
population = np.random.uniform(dom_l, dom_u, (npopulation, n_vars))
fitness = evaluate(population)
global_min = np.min(fitness)
global_max = np.max(fitness)

# Genetic Algorithm Loop
for generation in range(1, gens + 1):
    selected = tournament_selection(population, fitness)
    offspring = []
    for i in range(0, npopulation, 2):
        parent1, parent2 = selected[i], selected[i+1]
        child1, child2 = crossover(parent1, parent2)
        offspring.extend([child1, child2])
    offspring = np.array(offspring)[:npopulation]
    offspring = np.array([mutate2(child) for child in offspring])
    offspring_fitness = evaluate(offspring)
    
    # # Replacement using the Doomsday function
    # population, fitness = doomsday(offspring, offspring_fitness)

    # Replacement: Elitism (keep the best individual) (Could maybe change this)
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



# Normalize
# for generation in range(1, gens + 1):
#     # Normalize fitness values for selection
#     fitness = normalize(fitness, global_min, global_max)
    
#     selected = tournament_selection(population, fitness)
#     offspring = []
#     for i in range(0, npopulation, 2):
#         parent1, parent2 = selected[i], selected[i+1]
#         child1, child2 = crossover(parent1, parent2)
#         offspring.extend([child1, child2])
#     offspring = np.array(offspring)[:npopulation]
#     offspring = np.array([mutate2(child) for child in offspring])
#     offspring_fitness = evaluate(offspring)
    
#     # Update global min and max based on new fitness values
#     global_min = min(global_min, np.min(offspring_fitness))
#     global_max = max(global_max, np.max(offspring_fitness))

#     # Replace with Doomsday function or another appropriate function
#     population, fitness = doomsday(offspring, offspring_fitness)

#     # Normalize for logging purposes
#     best_fitness = np.max(fitness)
#     mean_fitness = np.mean(fitness)
#     std_fitness = np.std(fitness)
#     print(f'Generation {generation}: Best Fitness = {best_fitness:.4f}, Mean Fitness = {mean_fitness:.4f}, Std Fitness = {std_fitness:.4f}')

# # Save Best Solution
# best_idx = np.argmax(fitness)
# best_solution = population[best_idx]
# np.savetxt(os.path.join(experiment_name, 'best_solution.txt'), best_solution)
# print(f'Best solution saved with fitness: {fitness[best_idx]:.4f}')


