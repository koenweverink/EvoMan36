import os
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from evoman.environment import Environment
from controller1 import player_controller

def simulate(env, x):
    fitness, player_life, enemy_life, time = env.play(pcont=x)
    return fitness

def evaluate(env, x):
    return simulate(env, x)

def tournament_selection(population, fitness, k=3):
    selected = []
    pop_size = len(population)
    for _ in range(pop_size):
        candidates = np.random.randint(0, pop_size, k)
        selected.append(population[candidates[np.argmax(fitness[candidates])]])
    return np.array(selected)

def crossover(parent1, parent2):
    point = np.random.randint(0, len(parent1))
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2

def mutation(child, mutation_rate):
    mask = np.random.rand(*child.shape) < mutation_rate
    child[mask] += np.random.normal(0, 0.1, np.sum(mask))
    return np.clip(child, -1, 1)

def run_algorithm_a(env, npopulation, gens, mutation_rate):
    n_vars = (env.get_num_sensors() + 1) * 10 + (11) * 5 + (6) * 5
    population = np.random.uniform(-1, 1, (npopulation, n_vars))
    
    for generation in range(gens):
        fitness = np.array([evaluate(env, ind) for ind in population])
        
        selected = tournament_selection(population, fitness)
        
        offspring = []
        for i in range(0, npopulation - 1, 2):
            parent1, parent2 = selected[i], selected[i+1]
            child1, child2 = crossover(parent1, parent2)
            offspring.extend([mutation(child1, mutation_rate), mutation(child2, mutation_rate)])
        
        if npopulation % 2 != 0:
            offspring.append(mutation(selected[-1], mutation_rate))
        
        population = np.array(offspring)
        
        best_fitness = np.max(fitness)
        mean_fitness = np.mean(fitness)
        
    return [mean_fitness], [best_fitness], population[np.argmax(fitness)]

def objective(params, env):
    params['npopulation'] = int(params['npopulation'])
    params['gens'] = int(params['gens'])
    
    history_mean, history_max, solution = run_algorithm_a(env, **params)
    
    return {'loss': -history_max[-1], 'status': STATUS_OK}

def hyperopt_optimization(env, max_evals=20):
    space = {
        'npopulation': hp.quniform('npopulation', 50, 200, 1),
        'gens': hp.quniform('gens', 10, 30, 1),
        'mutation_rate': hp.uniform('mutation_rate', 0.01, 0.2),
    }

    trials = Trials()
    best = fmin(fn=lambda params: objective(params, env),
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)

    best['npopulation'] = int(best['npopulation'])
    best['gens'] = int(best['gens'])

    return best

def main():
    enemies = [6]  # List of enemies to train against
    n_hidden_neurons_1 = 10
    n_hidden_neurons_2 = 5

    for enemy in enemies:
        print(f"Optimizing parameters for enemy {enemy}")
        
        experiment_name = f'optimization_enemy_{enemy}'
        os.makedirs(experiment_name, exist_ok=True)
        
        env = Environment(
            experiment_name=experiment_name,
            enemies=[enemy],
            playermode="ai",
            player_controller=player_controller(n_hidden_neurons_1, n_hidden_neurons_2),
            enemymode="static",
            level=2,
            speed="fastest",
            visuals=False
        )
        
        best_params = hyperopt_optimization(env)
        
        print(f"Best parameters for enemy {enemy}: {best_params}")
        print("--------------------")

if __name__ == "__main__":
    main()