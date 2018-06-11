"""
This script reads the file with the stored results and shows on the console:
- Clock time and CPU time required for training
- Number of entries in a player's Q-table and how many entries were updated yet
- Number of games won / lost / drawn
- Total reward of both players
- Average reward of both players
- Average number of moves of both players
The cumulative reward over time is shown in a graph.
"""
__author__ = "Claudia Kutter"

import json
import collections as coll
import matplotlib.pyplot as plt

# SPECIFY FILES HERE
file = 'ttt_results_20000_SARSA-vs-SARSA.json'
q_table = 'ttt_table_X_SARSA_20000.json'

# Read in files
with open(file) as f:
	results = json.load(f)
with open(q_table) as f:
	table = json.load(f)

# Convert episode keys to int
results = {int(k): v for k, v in results.items()}
# Sort
results = coll.OrderedDict(sorted(results.items()))

# Data variables
length = len(results) - 1
episodes = list(results.keys())
episodes.pop(0)
win_x = 0
win_o = 0
draw = 0
total_reward_x = 0
total_reward_o = 0
moves_x = 0
moves_o = 0
rewards_x = []
rewards_o = []

# Extract data
for key, value in results.items():
	if key == -1:
		print("Elapsed time: {} sec".format(value['clock-time']))
		print("CPU processing time: {} sec".format(value['cpu-time']))
		print("*" * 50)
	if key >= 0:
		if value['winner'] == 1:
			win_x += 1
		if value['winner'] == -1:
			win_o += 1
		if value['winner'] == 0:
			draw += 1
		moves_x += value['X']['moves']
		moves_o += value['O']['moves']
		rewards_x.append(value['X']['reward'])
		rewards_o.append(value['O']['reward'])
	if key == length - 1:
		total_reward_x = value['X']['reward']
		total_reward_o = value['O']['reward']

# Count updated entries in Q-table
init_value = 0
for item in table.values():
	if item == 0.1:
		init_value += 1

# Print results to console
print("Entries in Q-table: ", len(table))
print("Initialized entries: ", init_value)
print("Updated entries: ", len(table) - init_value)
print("*" * 50)
print("Player X won: ", win_x)
print("Player O won: ", win_o)
print("Draw: ", draw)
print("Total reward of Player X after {} games: {}".format(length, total_reward_x))
print("Average reward of Player X: ", total_reward_x / length)
print("Average moves of Player X: ", moves_x / length)
print("Total reward of Player O after {} games: {}".format(length, total_reward_o))
print("Average reward of Player O: ", total_reward_o / length)
print("Average moves of Player O: ", moves_o / length)

# Plot cumulative reward over time
plt.plot(episodes, rewards_x, label='Player X', color='orange')
plt.plot(episodes, rewards_o, label='Player O', color='blue')
plt.ylabel('Cumulative reward')
plt.xlabel('Episode')
axes = plt.gca()
axes.set_ylim([-10000, 40000])
plt.legend(loc="upper left")
plt.show()
