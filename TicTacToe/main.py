"""
Starts playing the game Tic Tac Toe with the given agents. Either random agents or agents using the Q-learning or
SARSA algorithm can be used.
When playing with random agents still a learning rate and exploration rate is needed, however, it has no consequence
regarding the moves which the agent selects.
"""
import numpy as np
from time import process_time, perf_counter
from TicTacToe.game import *
from TicTacToe.agents import *

__author__ = "Claudia Kutter"
__license__ = "GPL"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

# Start application
if __name__ == '__main__':
	EPISODES = 100  # Number of episodes to play
	epsilon_max = 0.8  # Starting value for exploration
	epsilon_decay = np.linspace(0, epsilon_max, num=EPISODES)
	learning_rate_max = 0.5  # Starting value for learning rate
	learning_rate_decay = np.linspace(0, learning_rate_max, num=EPISODES)
	game_results = {}  # Storage for game results

	# Initialize game and players; set agents to True if they are to be trained
	player1 = SarsaAgent('X', True)
	player2 = SarsaAgent('O', True)
	game = TicTacToeGame(player1, player2)

	# Start measuring time
	clock_start = perf_counter()
	cpu_start = process_time()

	# Start playing episodes
	for episode in range(EPISODES):
		alpha = learning_rate_max - learning_rate_decay[episode]
		epsilon = epsilon_max - epsilon_decay[episode]
		result = game.play_game(epsilon=epsilon, alpha=alpha)
		# Store results
		game_results[str(episode + 1)] = result
		print("Episode {} finished after {} moves of player X".format(episode, result["X"]["moves"]))

	# Store results
	cpu_time = process_time() - cpu_start
	clock_time = perf_counter() - clock_start
	game_results['-1'] = {"cpu-time": cpu_time,
						  "clock-time": clock_time}
	# Write all results into a file
	with open("training_results/ttt_results_{}_{}-vs-{}.json".format(EPISODES, player1.method, player2.method),
			  "w") as file:
		file.write(json.dumps(game_results, indent=3, sort_keys=True))
	# Write q-tables of players into a file
	with open("training_results/ttt_table_{}_{}_{}.json".format(player1.mark, player1.method, EPISODES), "w") as file:
		file.write(player1.print_qtable())
	with open("training_results/ttt_table_{}_{}_{}.json".format(player2.mark, player2.method, EPISODES), "w") as file:
		file.write(player2.print_qtable())

	print("Training finished. Game results saved")
