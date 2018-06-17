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
file = '10000eps/othello_results_10000.json'

# Read in files
with open(file) as f:
	results = json.load(f)

# Convert episode keys to int
results = {int(k): v for k, v in results.items()}
# Sort
results = coll.OrderedDict(sorted(results.items()))

# Data variables
length = len(results) - 1
episodes = list(results.keys())
episodes.pop(0)
win_b = 0
win_w = 0
no_winner = 0
total_reward_b = 0
total_reward_w = 0
total_moves_b = 0
total_moves_w = 0
moves_b = []
moves_w = []
rewards_b = []
rewards_w = []

# Extract data
for key, value in results.items():
	if key == -1:
		print("Elapsed time: {} sec".format(value['clock-time']))
		print("CPU processing time: {} sec".format(value['cpu-time']))
		print("*" * 50)
	if key >= 0:
		if value['winner'] == 1:
			win_b += 1
		if value['winner'] == -1:
			win_w += 1
		if value['winner'] == -2:
			no_winner += 1
		total_moves_b += value['black']['moves']
		total_moves_w += value['white']['moves']
		moves_b.append(value['black']['moves'])
		moves_w.append(value['white']['moves'])
		rewards_b.append(value['black']['reward'])
		rewards_w.append(value['white']['reward'])
	if key == length - 1:
		total_reward_b = value['black']['reward']
		total_reward_w = value['white']['reward']

# Print results to console
print("Player Black won: ", win_b)
print("Player White won: ", win_w)
print("No winner: ", no_winner)
print("*" * 50)
print("Total reward of Player Black after {} games: {}".format(length, total_reward_b))
print("Average reward of Player Black: ", total_reward_b / length)
print("Average moves of Player Black: ", total_moves_b / length)
print("Total reward of Player White after {} games: {}".format(length, total_reward_w))
print("Average reward of Player White: ", total_reward_w / length)
print("Average moves of Player White: ", total_moves_w / length)

# Plot moves over time
plt.plot(episodes, moves_b, label='Player Black', color='orange')
plt.ylabel('Number of moves')
plt.xlabel('Episode')
axes = plt.gca()
axes.set_ylim([0, 35])
plt.legend(loc="upper left")
plt.show()

"""
# Plot cumulative reward over time
plt.plot(episodes, rewards_b, label='Player X', color='orange')
plt.plot(episodes, rewards_w, label='Player O', color='blue')
plt.ylabel('Cumulative reward')
plt.xlabel('Episode')
axes = plt.gca()
axes.set_ylim([0, 20000])
plt.legend(loc="upper left")
plt.show()
"""
