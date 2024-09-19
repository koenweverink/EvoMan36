import os
import numpy as np
from evoman.environment import Environment
from controller1 import player_controller
from evoman.controller import Controller

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
npopulation = 100       # populationsize
gens = 30               # Number of generations
mutation_rate = 0.05    # Mutation rate
dom_u, dom_l = 1, -1    # Upper and lower bounds of the weights

# Number of variables in the controller
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5


def sigmoid_activation(x):
	return 1./(1.+np.exp(-x))

# implements controller structure for player
class player_controller(Controller):
	def __init__(self, _n_hidden):
		self.n_hidden = [_n_hidden]

	def set(self,controller, n_inputs):
		# Number of hidden neurons

		if self.n_hidden[0] > 0:
			# Preparing the weights and biases from the controller of layer 1

			# Biases for the n hidden neurons
			self.bias1 = controller[:self.n_hidden[0]].reshape(1, self.n_hidden[0])
			# Weights for the connections from the inputs to the hidden nodes
			weights1_slice = n_inputs * self.n_hidden[0] + self.n_hidden[0]
			self.weights1 = controller[self.n_hidden[0]:weights1_slice].reshape((n_inputs, self.n_hidden[0]))

			# Outputs activation first layer.


			# Preparing the weights and biases from the controller of layer 2
			self.bias2 = controller[weights1_slice:weights1_slice + 5].reshape(1, 5)
			self.weights2 = controller[weights1_slice + 5:].reshape((self.n_hidden[0], 5))

	def control(self, inputs, controller):
		# Normalises the input using min-max scaling
		inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))

		if self.n_hidden[0]>0:
			# Preparing the weights and biases from the controller of layer 1

			# Outputs activation first layer.
			output1 = sigmoid_activation(inputs.dot(self.weights1) + self.bias1)

			# Outputting activated second layer. Each entry in the output is an action
			output = sigmoid_activation(output1.dot(self.weights2)+ self.bias2)[0]
		else:
			bias = controller[:5].reshape(1, 5)
			weights = controller[5:].reshape((len(inputs), 5))

			output = sigmoid_activation(inputs.dot(weights) + bias)[0]

		# takes decisions about sprite actions
		if output[0] > 0.5:
			left = 1
		else:
			left = 0

		if output[1] > 0.5:
			right = 1
		else:
			right = 0

		if output[2] > 0.5:
			jump = 1
		else:
			jump = 0

		if output[3] > 0.5:
			shoot = 1
		else:
			shoot = 0

		if output[4] > 0.5:
			release = 1
		else:
			release = 0

		return [left, right, jump, shoot, release]
	
# implements controller structure for enemy
class enemy_controller(Controller):
	def __init__(self, _n_hidden):
		# Number of hidden neurons
		self.n_hidden = [_n_hidden]

	def control(self, inputs,controller):
		# Normalises the input using min-max scaling
		inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))

		if self.n_hidden[0]>0:
			# Preparing the weights and biases from the controller of layer 1

			# Biases for the n hidden neurons
			bias1 = controller[:self.n_hidden[0]].reshape(1,self.n_hidden[0])
			# Weights for the connections from the inputs to the hidden nodes
			weights1_slice = len(inputs)*self.n_hidden[0] + self.n_hidden[0]
			weights1 = controller[self.n_hidden[0]:weights1_slice].reshape((len(inputs),self.n_hidden[0]))

			# Outputs activation first layer.
			output1 = sigmoid_activation(inputs.dot(weights1) + bias1)

			# Preparing the weights and biases from the controller of layer 2
			bias2 = controller[weights1_slice:weights1_slice + 5].reshape(1,5)
			weights2 = controller[weights1_slice + 5:].reshape((self.n_hidden[0],5))

			# Outputting activated second layer. Each entry in the output is an action
			output = sigmoid_activation(output1.dot(weights2)+ bias2)[0]
		else:
			bias = controller[:5].reshape(1, 5)
			weights = controller[5:].reshape((len(inputs), 5))

			output = sigmoid_activation(inputs.dot(weights) + bias)[0]

		# takes decisions about sprite actions
		if output[0] > 0.5:
			attack1 = 1
		else:
			attack1 = 0

		if output[1] > 0.5:
			attack2 = 1
		else:
			attack2 = 0

		if output[2] > 0.5:
			attack3 = 1
		else:
			attack3 = 0

		if output[3] > 0.5:
			attack4 = 1
		else:
			attack4 = 0

		return [attack1, attack2, attack3, attack4]


# Run the sim and return the fitness
def simulate(x):
    f, _, _, _ = env.play(pcont=x)
    return f

# Evaluate the current population
def evaluate(population):
    return np.array([simulate(individual) for individual in population])

# Tournament Selection
def tournament_selection(population, fitness, k=2):
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
