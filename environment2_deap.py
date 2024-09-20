import os
import numpy as np
from deap import base, creator, tools
from evoman.environment import Environment
from controller2 import player_controller

# Configuration
experiment_name = 'deap_three_parent_crossover_experiment'
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
gens = 30              # Number of generations
mutation_rate = 0.2    # Mutation rate
dom_u, dom_l = 1, -1   # Upper and lower bounds of the weights

# Number of variables in the controller
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

# Create DEAP framework components
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

# DEAP toolbox
toolbox = base.Toolbox()

# Attribute generator: random initialization of the population
toolbox.register("attr_float", np.random.uniform, dom_l, dom_u)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_vars)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Run the sim and return the fitness
def simulate(individual):
    return env.play(pcont=individual)[0]

# Custom three-parent crossover
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

# Custom mutation
def mutate(individual):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] += np.random.normal(0, 0.1)
            individual[i] = np.clip(individual[i], dom_l, dom_u)
    return individual,

# Register DEAP operators
toolbox.register("evaluate", simulate)
toolbox.register("mate", three_parent_crossover)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

# Replacement with Elitism: Keep the best individual from the current generation
def elitism(population, fitnesses, offspring, offspring_fitnesses):
    best_idx = np.argmax(fitnesses)
    worst_idx = np.argmin(offspring_fitnesses)
    if fitnesses[best_idx] > offspring_fitnesses[worst_idx]:
        offspring[worst_idx] = population[best_idx]
        offspring_fitnesses[worst_idx] = fitnesses[best_idx]
    return offspring, offspring_fitnesses

# Main evolutionary algorithm
def main():
    population = toolbox.population(n=npopulation)
    
    # Evaluate the initial population
    fitnesses = [toolbox.evaluate(individual) for individual in population]
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = (fit,)

    # Begin the evolution
    for gen in range(gens):
        
        # Select parents using tournament selection
        offspring = []
        selected = toolbox.select(population, len(population))
        for i in range(0, len(selected), 3):
            parent1, parent2, parent3 = selected[i], selected[i+1], selected[i+2]
            child1, child2, child3 = toolbox.mate(parent1, parent2, parent3)
            offspring.extend([creator.Individual(child1), creator.Individual(child2), creator.Individual(child3)])
        offspring = offspring[:npopulation]
        
        # Mutate the offspring
        offspring = [toolbox.mutate(ind)[0] for ind in offspring]

        # Evaluate the offspring
        offspring_fitnesses = [toolbox.evaluate(ind) for ind in offspring]
        for ind, fit in zip(offspring, offspring_fitnesses):
            ind.fitness.values = (fit,)

        # Apply elitism
        population, fitnesses = elitism(population, fitnesses, offspring, offspring_fitnesses)

        # Logging
        best_fitness = np.max(fitnesses)
        mean_fitness = np.mean(fitnesses)
        std_fitness = np.std(fitnesses)
        
        print(f'Generation {gen + 1}, Best Fitness = {best_fitness:.4f}, Mean Fitness = {mean_fitness:.4f}, Std Fitness = {std_fitness:.4f}')

    # Save best solution
    best_ind = tools.selBest(population, 1)[0]
    np.savetxt(os.path.join(experiment_name, 'best_solution.txt'), best_ind)
    print(f'Best solution saved with fitness: {best_ind.fitness.values[0]:.4f}')

if __name__ == "__main__":
    main()
