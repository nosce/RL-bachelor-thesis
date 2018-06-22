"""
Starts playing the game Othello with the given agents. Either random agents or agents using a deep Q-network
can be used.
When playing with random agents still a exploration rate epsilon is needed, however, it has no consequence regarding
the moves which the agent selects.
"""
import json
from time import process_time, perf_counter
from Othello.game import *
from Othello.agents import *

__author__ = "Claudia Kutter"
__license__ = "GPL"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

# Start application
if __name__ == '__main__':
	EPISODES = 80000  # Number of episodes to play
	epsilon_max = 1.0  # Starting value for exploration
	epsilon_min = 0.05  # Lowest value for exploration
	epsilon_decay = 1 / (EPISODES // 32)  # Steps for decaying exploration
	game_results = {}  # Storage for game results

	# Initialize game and players; set DQN agents to True if they are to be trained
	# player1 = DQNAgent('black', True)
	# player2 = DQNAgent('white', True)
	player1 = QAgent('black', True)
	player2 = QAgent('white', True)
	game = OthelloGame(player1, player2)

	# Start measuring time
	clock_start = perf_counter()
	cpu_start = process_time()

	# Start playing episodes
	for episode in range(EPISODES):
		epsilon = max(epsilon_min, epsilon_max - episode * epsilon_decay)
		result = game.play_game(epsilon)
		# Store results
		game_results[str(episode + 1)] = result
		print("Episode {} finished after {} black moves".format(episode, result["black"]["moves"]))

		# Store results every 5,000 steps
		if (episode + 1) % 5000 == 0:
			cpu_time = process_time() - cpu_start
			clock_time = perf_counter() - clock_start
			game_results['-1'] = {"cpu-time": cpu_time,
								  "clock-time": clock_time}
			# Write all results into a file
			with open("othello_results_{}.json".format(episode), "w") as file:
				file.write(json.dumps(game_results, indent=3, sort_keys=True))
			# Reset storage to save space
			game_results.clear()
			print("********* Results stored *********")
	print("Training finished. Game results saved")
