"""
This file contains the implementation of different agents for Tic Tac Toe.
__Player: Serves as parent class for all agents and implements common functions such as storing rewards
__RandomAgent: This agents makes random moves based on all valid moves in a certain board state
__QAgent: This agent uses Q-learning to store and update Q-values for state-action-pairs. When initializing the agent,
			it can be specified whether the agent should be trained (True) or a stored Q-table should be loaded.
			The stored Q-table must be specified in a file: /training_results/qtable.json
__SarsaAgent: This agent uses SARSA to store and update Q-values for state-action-pairs. When initializing the agent,
			it can be specified whether the agent should be trained (True) or a stored Q-table should be loaded.
			The stored Q-table must be specified in a file: /training_results/sarsa-table.json
"""
# TODO add loading of q-table and sarsa-table
import numpy as np
import json

__author__ = "Claudia Kutter"
__license__ = "GPL"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

np.random.seed(42)  # For reproducibility


class Player(object):
	"""
	Represents a player in the game. The player X has ID 1, the player O has ID -1. The IDs are used to
	mark the position of the player's stones on the board.
	"""

	def __init__(self, mark):
		self.id = 1 if mark == 'X' else -1
		self.mark = mark
		self.reward = None
		self.total_reward = 0  # Reward accumulated during all games
		self.moves = -1  # Number of moves during the game; starts with -1 to correct final
		# move selection when game is already over
		self.last_state = None
		self.last_action = None
		self.qtable = {}  # Q-table for storing state-action-values

	def reset_for_new_game(self, epsilon, alpha):
		"""
		Resets variables for storing game states, moves and rewards when. Required when agent plays multiple episodes
		of the game in a row.
		:param epsilon: Exploration rate of the agent
		:param alpha: Learning rate of the agent
		:return: None
		"""
		self.last_state = None
		self.last_action = None
		self.reward = None
		self.moves = -1

	@staticmethod
	def possible_actions(state):
		"""
		Returns all fields where moves are possible because they are empty
		:param state: Current board state as tuple
		:return: Array of empty fields. Fields are given as array [row, column]
		"""
		actions = []
		for row in range(3):
			for col in range(3):
				if state[row][col] == 0:
					actions.append((row, col))
		return actions

	def store_reward(self, reward):
		"""
		Stores the numerical reward that the agent receives after making a move and adds it to the total reward that
		the agents collects during a game
		:param reward: Numerical reward to be stored
		:return: None
		"""
		self.reward = reward
		self.total_reward += reward

	def get_qvalue(self, state, action):
		"""
		Returns the Q-value for a state-action pair if it exists, otherwise the value is initialized with 0.1
		in order to encourage exploration of new states
		:param state: Board state as a tuple
		:param action: Action as tuple (row, column)
		:return: Q-value for state-action pair
		"""
		if (state, action) not in self.qtable:
			self.qtable[(state, action)] = 0.1
		return self.qtable[(state, action)]

	def print_qtable(self):
		"""
		Returns the Q-table in json-format, e.g in order to store it in a file
		:return: Q-table in json-format
		"""
		json_table = {}
		for key, value in self.qtable.items():
			json_table[str(key)] = value
		return json.dumps(json_table, indent=3, sort_keys=True)

	def learn(self, state):
		pass

	def select_action(self, state):
		pass


class QAgent(Player):
	"""
	Agent using Q-learning for learning and selecting game moves
	"""

	def __init__(self, mark):
		Player.__init__(self, mark)
		self.method = "QL"
		self.epsilon = 0.5  # chance of exploration instead of exploitation
		self.alpha = 0.8  # learning rate
		self.gamma = 0.9  # discount factor for future rewards

	def reset_for_new_game(self, epsilon, alpha):
		"""
		Resets variables for storing game states, moves and rewards. Required when agent plays multiple episodes
		of the game in a row.
		:param epsilon: Exploration rate of the agent
		:param alpha: Learning rate of the agent
		:return: None
		"""
		self.last_state = None
		self.last_action = None
		self.reward = None
		self.moves = 0
		self.epsilon = epsilon
		self.alpha = alpha

	def learn(self, state):
		"""
		Initiates the learning process by updating the Q-table and selecting the next action
		:param state: Current board state as a tuple
		:return: Selected action as tuple (row, column)
		"""
		self.update_qtable(state)
		return self.select_action(state)

	def select_action(self, board_state):
		"""
		Stores the board state and selects an action epsilon greedily
		:param board_state: Current board state as a tuple
		:return: Selected action as tuple (row, column)
		"""
		# Store board state
		self.last_state = board_state
		# Randomly choose a number between 0 and 1; if it is lower than epsilon, explore possible actions randomly
		if len(self.possible_actions(board_state)) == 0:
			return None
		if np.random.random() < self.epsilon:
			actions = self.possible_actions(board_state)
			self.last_action = actions[np.random.choice(len(actions))]
		else:
			# Get the Q-values for all possible actions in the current state
			all_qvalues = {}
			for action in self.possible_actions(board_state):
				qvalue = self.get_qvalue(board_state, action)
				# Actions are stored by Q-values; some actions might have the same Q-value
				if qvalue not in all_qvalues:
					all_qvalues[qvalue] = [action]
				else:
					all_qvalues[qvalue].append(action)
			# Find the highest Q-value and how many actions are linked to it
			max_qvalue = max(all_qvalues)
			best_actions = len(all_qvalues[max_qvalue])
			# If there is more than one best action, choose randomly
			if best_actions > 1:
				move = all_qvalues[max_qvalue][np.random.choice(best_actions)]
			else:
				move = all_qvalues[max_qvalue][0]
			self.last_action = move
		self.moves += 1
		return self.last_action

	def update_qtable(self, new_state):
		"""
		Recalculates Q-values if values are available, i.e. after one state-action-transition
		:param new_state: Following state after taking an action
		:return: None
		"""
		if self.last_state and self.last_action and not (self.reward is None):
			current_value = self.get_qvalue(self.last_state, self.last_action)
			possible_returns = [self.get_qvalue(new_state, a) for a in self.possible_actions(new_state)]
			max_return = max(possible_returns) if possible_returns else 0
			self.qtable[(self.last_state, self.last_action)] = current_value + self.alpha * (self.reward + self.gamma *
																							 max_return - current_value)


class SarsaAgent(Player):
	"""
	Agent using SARSA for learning and selecting game moves
	"""

	def __init__(self, mark):
		Player.__init__(self, mark)
		self.method = "SARSA"
		self.epsilon = 0.5  # chance of exploration instead of exploitation
		self.alpha = 0.8  # learning rate
		self.gamma = 0.9  # discount factor for future rewards
		self.current_state = None
		self.current_action = None

	def reset_for_new_game(self, epsilon, alpha):
		"""
		Resets variables for storing game states, moves and rewards. Required when agent plays multiple episodes
		of the game in a row.
		:param epsilon: Exploration rate of the agent
		:param alpha: Learning rate of the agent
		:return: None
		"""
		self.current_state = None
		self.current_action = None
		self.last_state = None
		self.last_action = None
		self.reward = None
		self.moves = 0
		self.epsilon = epsilon
		self.alpha = alpha

	def learn(self, state):
		"""
		Initiates the learning process by updating the Q-table and selecting the next action
		:param state: Current board state as a tuple
		:return: Selected action as tuple (row, column)
		"""
		action = self.select_action(state)
		self.update_qtable()
		return action

	def select_action(self, board_state):
		"""
		Stores the board state and selects an action epsilon greedily
		:param board_state: Current board state as a tuple
		:return: Selected action as tuple (row, column)
		"""
		self.last_state = self.current_state
		self.current_state = board_state
		self.last_action = self.current_action
		# Randomly choose a number between 0.0 and 1.0; if it is lower than epsilon, explore possible actions randomly
		if len(self.possible_actions(board_state)) == 0:
			return None
		if np.random.random() < self.epsilon:
			actions = self.possible_actions(board_state)
			self.current_action = actions[np.random.choice(len(actions))]
		else:
			# Get the Q-values for all possible actions in the current state
			all_qvalues = {}
			for action in self.possible_actions(board_state):
				qvalue = self.get_qvalue(board_state, action)
				# Actions are stored by Q-values; some actions might have the same Q-value
				if qvalue not in all_qvalues:
					all_qvalues[qvalue] = [action]
				else:
					all_qvalues[qvalue].append(action)
			# Find the highest Q-value and how many actions are linked to it
			max_qvalue = max(all_qvalues)
			best_actions = len(all_qvalues[max_qvalue])
			# If there is more than one best action, choose randomly
			if best_actions > 1:
				move = all_qvalues[max_qvalue][np.random.choice(best_actions)]
			else:
				move = all_qvalues[max_qvalue][0]
			self.current_action = move
		self.moves += 1
		return self.current_action

	def update_qtable(self):
		"""
		Recalculates Q-values if values are available, i.e. after one state-action-transition
		:return: None
		"""
		if self.last_state and self.last_action and not (self.reward is None):
			current_value = self.get_qvalue(self.last_state, self.last_action)
			next_return = self.get_qvalue(self.current_state, self.current_action)
			self.qtable[(self.last_state, self.last_action)] = current_value + self.alpha * \
															   (self.reward + self.gamma * next_return - current_value)


class RandomAgent(Player):
	"""
	Agent making random moves
	"""

	def __init__(self, mark):
		Player.__init__(self, mark)
		self.method = "Random"

	def learn(self, state):
		"""
		Starts the learning process (not relevant for random player) and returns the next action
		:param state: Current board state
		:return: Selected move as a tuple for the position of the field; (-1,-1) if no valid move is available
		"""
		return self.select_action(state)

	def select_action(self, state):
		"""
		Selects a random action out of all given valid actions
		:param state: Current board state
		:return: Selected move as a tuple for the position of the field; (-1,-1) if no valid move is available
		"""
		self.moves += 1
		actions = self.possible_actions(state)
		if len(actions) > 0:
			return tuple(actions[np.random.choice(len(actions))])
		else:
			return -1, -1
