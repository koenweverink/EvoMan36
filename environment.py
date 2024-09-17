# imports framework
import sys, os

from evoman.environment import Environment
from controller1 import player_controller
# from group36.controller2 import player_controller		# Uncomment this line and comment the line above to test the second controller

# imports other libs
import numpy as np

experiment_name = 'controller_specialist_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
n_hidden_neurons = 0

# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
				  playermode="ai",
				  player_controller=player_controller(n_hidden_neurons),
			  	  speed="normal",
				  enemymode="static",
				  level=2,
				  visuals=True)


# tests saved demo solutions for each enemy
for en in range(1, 9):

	#Update the enemy
	env.update_parameter('enemies',[en])

	# Load specialist controller
	sol = np.loadtxt('solutions_demo/demo_'+str(en)+'.txt')
	print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY '+str(en)+' \n')
	env.play(sol)
