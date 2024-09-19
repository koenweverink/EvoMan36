import sys,os
import random
import array
import time

import numpy as np
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

sys.path.insert(0, 'evoman') 
from environment import Environment
from controller1 import player_controller

os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

hidden_units_num=10
chorosome_len = (20+1) * hidden_units_num + (hidden_units_num+1)*5


upper_limit=1
lower_limit=-1
population_num=100
max_generation=30

def limit_bound(gene):
    if gene>upper_limit:
        return upper_limit
    elif gene<lower_limit:
        return lower_limit
    else:
        return gene

def experiment_setting():
    crossover_rate=0.8
    F=1
    enemies=[1,2,3]
    rounds_per_enemy=10
    return crossover_rate,F,enemies,rounds_per_enemy


def select_parents(population,population_fitness):
    a,b,c = toolbox.select(population)
    return a,b,c

#replace
def select_survivors(pop,pop_fitness,prev_pop_num):
    #check num equality?
    pop_cp=pop[:prev_pop_num]
    pop_fitness_cp=pop_fitness[:prev_pop_num]
    offspring=pop[prev_pop_num:]
    offspring_fitness=pop_fitness[prev_pop_num:]
    for i,f in enumerate(offspring_fitness):
        if f>pop_fitness[i]:
            pop_cp[i]=offspring[i]
            pop_fitness_cp[i]=offspring_fitness[i]
    return pop_cp,pop_fitness_cp

#DE crossover, produce an array
def crossover(a,b,c,offspring,F,crossover_rate,isMutation=False):
    y = toolbox.clone(offspring)
    index = random.randrange(chorosome_len) #ensure at least one crossover
    for i,value in enumerate(offspring):
        if i == index or random.random() < crossover_rate:
            step = a[i] + F*(b[i]-c[i]) #use best solution as the base vector0
            if isMutation and random.random() < 0.2:
                step=step+np.random.normal(0,1)
            y[i]=limit_bound(step)   
    return [y] 

def evaluate(env,population):
    return np.array(list(map(lambda y: env.play(pcont=np.array(y))[0], population)))

def experiment(crossover_rate,F,enemy,round):
    
    '''
        evaluate first generation
    '''
    experiment_name = 'exp_enemy_'+str(enemy)+'_round_'+str(round)
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    
    env = Environment(experiment_name=experiment_name,
                  enemies=enemy,
                  playermode="ai",
                  player_controller=player_controller(hidden_units_num),
                  enemymode="static",
                  level=2,
                  logs='off',
                  speed="fastest")
    env.state_to_log() # checks environment state
    population=toolbox.population(n=population_num)
    population_fitness=toolbox.evaluate(env,population)
    notimproved=0
    current_iter=0
    solutions=[population,population_fitness]
    ini = time.time()  # sets time marker
    env.update_solutions(solutions)
    best=stat_and_record(experiment_name,current_iter,population,population_fitness)
    current_best_fit=population_fitness[best]
    for g in range(1,max_generation):
        for k in range(len(population)):
            a,b,c=select_parents(population,population_fitness)
            if notimproved>=10:
                offspring=crossover(a,b,c,population[k],F,crossover_rate,True)
            else:
                offspring=crossover(a,b,c,population[k],F,crossover_rate)
            fit_cross=toolbox.evaluate(env,offspring)
            population.extend(offspring)
            population_fitness=np.append(population_fitness,fit_cross)
            # two array  how to init for them
        population,population_fitness=select_survivors(population,population_fitness,population_num)
        best = np.argmax(population_fitness) #best solution in generation
        population_fitness[best] = float(evaluate(env,np.array([population[best] ]))[0]) # repeats best eval, for stability issues
        
        if population_fitness[best] <= current_best_fit:
            notimproved += 1
        else:
            current_best_fit = population_fitness[best]
            notimproved = 0

        if notimproved >= 15:
            file_aux  = open(experiment_name+'/results.txt','a')
            file_aux.write('\n train over')
            file_aux.close()
            solutions = [population, population_fitness]
            env.update_solutions(solutions)
            env.save_state()
            return

        if notimproved >= 10:
            file_aux  = open(experiment_name+'/results.txt','a')
            file_aux.write('\n start mutation')
            file_aux.close()

        current_best_fit = population_fitness[best]
        stat_and_record(experiment_name,g,population,population_fitness)
        # saves simulation state
        solutions = [population, population_fitness]
        env.update_solutions(solutions)
        env.save_state()
    return



def stat_and_record(exp_name,curr_iter,pops,fitnesses):
    best=np.argmax(fitnesses)
    mean=np.mean(fitnesses)
    std=np.std(fitnesses)
    file_aux  = open(exp_name+'/results.txt','a')
    file_aux.write('\n'+str(curr_iter)+'\t'+str(round(fitnesses[best],6))+'\t'+str(round(mean,6))+'\t'+str(round(std,6))   )
    file_aux.close()
    # saves generation number
    file_aux  = open(exp_name+'/gen.txt','w')
    file_aux.write(str(curr_iter))
    file_aux.close()
    # saves file with the best solution
    np.savetxt(exp_name+'/best.txt',pops[best])
    return best


creator.create("FitnessMax", base.Fitness, weights=(1.0,))  #fitness should be modified, what about weights
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, lower_limit, upper_limit) #initializing first generation
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, chorosome_len)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("select", tools.selRandom, k=3) #select by probs as demo
toolbox.register("evaluate",evaluate)

crossover_rate,F,enemies,rounds_per_enemy=experiment_setting()
for e in [1,2,3]:
    for r in range(rounds_per_enemy):
        experiment(crossover_rate,F,[e],r)