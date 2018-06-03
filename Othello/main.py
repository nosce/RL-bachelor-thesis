"""
Starts playing the game Othello with the given agents. Either random agents or agents using a deep Q-network
can be used.
When playing with random agents still a learning rate epsilon is needed, however, it has no consequence regarding
the moves which the agent selects.
"""
import json
from time import process_time, perf_counter
from Othello.game import *

__author__ = "Claudia Kutter"
__license__ = "GPL"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

# Start application
if __name__ == '__main__':
	EPISODES = 1000  # Number of episodes to play
	epsilon_start = 1.0  # Starting value for exploration
	epsilon_end = 0.05  # Lowest value for exploration
	epsilon_decay = np.linspace(0, epsilon_start - epsilon_end, num=EPISODES)  # Steps for decaying exploration
	game_results = {}  # Storage for game results

	# Initialize game and players; set DQN agents to True if they are to be trained
	player1 = DQNAgent('black', True)
	player2 = DQNAgent('white', True)
	game = OthelloGame(player1, player2)

	# Start measuring time
	clock_start = perf_counter()
	cpu_start = process_time()

	# Start playing episodes
	for episode in range(EPISODES):
		epsilon = epsilon_start - epsilon_decay[episode]
		result = game.play_game(epsilon)
		# Store results
		game_results[str(episode + 1)] = result
		print("Episode {} finished after {} black moves".format(episode, result["black"]["moves"]))

	# Store results
	cpu_time = process_time() - cpu_start
	clock_time = perf_counter() - clock_start
	game_results["cpu-time"] = cpu_time
	game_results["clock-time"] = clock_time
	# Write all results into a file
	with open("training_results/othello_results_{}.json".format(EPISODES), "w") as file:
		file.write(json.dumps(game_results, indent=3, sort_keys=True))
	print("Training finished. Game results saved")
