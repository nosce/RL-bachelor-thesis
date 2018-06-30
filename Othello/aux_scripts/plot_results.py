"""
This script reads the file with the stored results and shows on the console:
- Clock time and CPU time required for training
- Number of games won / lost / drawn
- Total reward of both players
- Average reward of both players
- Average number of moves of both players
The number of moves and clock/CPU time is output as a graph showing the results over time.
"""
__author__ = "Claudia Kutter"

import json
import collections as coll
import matplotlib.pyplot as plt

# ************ SPECIFY FILE HERE ************ #
file = 'othello-results_m500b64.json'

# Read in file
with open(file) as f:
	results = json.load(f)

# Convert episode keys to int
results = {int(k): v for k, v in results.items()}
# Sort
results = coll.OrderedDict(sorted(results.items()))

# Data variables
episodes = []
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
cpu_times = []
training_times = []
record_steps = []

# Extract data
for key, value in results.items():
	if key < 0:
		record_steps.append(key * (-5000))
		cpu_times.append(value['cpu-time'])
		training_times.append(value['clock-time'])
	if key >= 0:
		episodes.append(key)
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
length = len(episodes)

# Print results to console
print("Player Black won: ", win_b)
print("Player White won: ", win_w)
print("No winner: ", no_winner)
print("*" * 50)
print("Total reward of Player Black after {} games: {}".format(length, total_reward_b))
print("Average reward of Player Black: ", total_reward_b / length)
print("Average moves of Player Black: ", total_moves_b / length)
print("Total moves of Player Black: ", total_moves_b)
print("Total reward of Player White after {} games: {}".format(length, total_reward_w))
print("Average reward of Player White: ", total_reward_w / length)
print("Average moves of Player White: ", total_moves_w / length)

# Name of plotted files
name = file[:-5] if file.endswith('.json') else file
# Plot moves over time
plt.figure(figsize=(20, 10))
plt.plot(episodes, moves_b, label='Player Black', color='#8bb174', linewidth=0.3)
plt.ylabel('Number of moves')
plt.xlabel('Episode')
axes = plt.gca()
axes.set_ylim([0, 35])
plt.legend(loc="upper left")
plt.savefig('{}_moves.png'.format(name))
plt.close()

# Plot time consumption
plt.figure(figsize=(20, 10))
plt.plot(record_steps, cpu_times, label='CPU time', color='#b2463c')
plt.plot(record_steps, training_times, label='Clock time', color='#177e88')
plt.ylabel('Seconds')
plt.xlabel('Episode')
axes = plt.gca()
axes.set_ylim([0, 800000])
plt.legend(loc="upper left")
plt.savefig('{}_time.png'.format(name))
plt.close()
