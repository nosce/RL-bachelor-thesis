"""
This file contains the implementation of different agents for Othello.
__Player: Serves as parent class for all agents and implements common functions such as storing rewards
__RandomAgent: This agents makes random moves based on all valid moves in a certain board state
__DQNAgent: This agent uses a deep Q-network to estimate the next best move. When initializing the agent,
			it can be specified whether the agent should be trained (True) or initialize its network with given weights
			which must be specified in a file: /training_results/final_weights_<colour>.h5 with <colour> being the
			colour of the agent (black/white)
"""

import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from collections import deque

__author__ = "Claudia Kutter"
__license__ = "GPL"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

random.seed(42)  # For reproducibility
np.random.seed(42)

class Player(object):
	"""
	Represents a player in the game. The black player's ID is 1, the white player's ID is -1, The IDs are used to
	mark the position of the player's stones on the board.
	"""

	def __init__(self, colour):
		self.id = 1 if colour == 'black' else -1
		self.colour = colour
		self.valid_moves = []
		self.reward = None
		self.total_reward = 0  # Reward accumulated during the game
		self.moves = 0  # Number of moves during the game
		self.total_moves = 0  # Total number of moves in all episodes
		self.invalid_moves = 0  # Number of invalid moves during the game

	def reset_for_new_game(self, eps):
		"""
		Resets variables for storing game states, moves and rewards when. Required when agent plays multiple episodes
		of the game in a row.
		:param eps: Exploration rate of the agent
		:return: None
		"""
		self.reward = None
		self.moves = 0
		self.invalid_moves = 0

	def get_valid_moves(self, state):
		"""
		Checks whether there are any valid moves for the player on the board. A move is considered valid if it leads
		to stones of the opponents being flipped.
		:param state: Current board object
		:return: Array with all valid moves for the player
		"""
		self.valid_moves.clear()
		empty_fields = np.transpose(np.where(state.board == 0))
		for field in empty_fields:
			if state.find_flanks(field, self.id, False):
				self.valid_moves.append(field.tolist())
		state.draw_valid_moves(self.valid_moves)
		# Log whether there is any valid move for the player
		state.no_moves_possible[self.id] = True if len(self.valid_moves) == 0 else False
		return self.valid_moves

	def store_reward(self, reward):
		"""
		Stores the numerical reward that the agent receives after making a move and adds it to the total reward that
		the agents collects during a game
		:param reward: Numerical reward to be stored
		:return: None
		"""
		self.reward = reward
		self.total_reward += reward

	def learn(self, new_state, done):
		pass

	def select_action(self, state):
		pass


class DQNAgent(Player):
	"""
	Agent using a Deep Q-Network (DQN) for learning and selecting game moves
	"""
	def __init__(self, colour, train):
		Player.__init__(self, colour)
		self.last_state = None
		self.last_action = None
		self.gamma = 0.9  # Weight for future rewards
		self.epsilon = 1.0  # Exploration rate will be set during resetting
		self.memory = deque(maxlen=2000)  # Sets capacity of replay memory D
		self.training_model = self.setup_network()  # Network for playing (Q)
		self.target_model = self.setup_network()  # Network to be trained (Q^)
		self.c = 300  # Rate at which the target network is reset
		self.train = train  # Whether agent should be trained or only use its current knowledge
		if not self.train:
			self.training_model.load_weights('training_results/final_weights_{}.h5')
			self.training_model.compile(loss='mean_squared_error', optimizer='sgd')

	def setup_network(self):
		"""
		Sets up the layers of the neural network
		:return: Model of the neural network
		"""
		model = Sequential()
		model.add(Flatten(input_shape=(8, 8, 2)))
		model.add(Dense(256, activation='relu'))
		model.add(Dense(128, activation='relu'))
		model.add(Dense(128, activation='relu'))
		model.add(Dense(64, activation='linear'))
		model.compile(loss='mse', optimizer='sgd')
		return model

	def reset_for_new_game(self, eps):
		"""
		Resets variables for storing game states, moves and rewards. Required when agent plays multiple episodes
		of the game in a row.
		:param eps: Exploration rate of the agent
		:return: None
		"""
		self.epsilon = eps
		self.last_state = None
		self.last_action = None
		self.reward = None
		self.moves = 0

	def select_action(self, state):
		"""
		Selects a move from all available valid moves. With probability epsilon a random move is selected else the
		move with the highest Q-value predicted by the training network is selected
		:param state: Array of current board state
		:return: Selected move as a tuple for the position of the field; (-1,-1) if no valid move is possible
		"""
		self.last_state = state
		self.moves += 1
		self.total_moves += 1
		# Makes a random move
		if len(self.valid_moves) > 0:
			# Select random move with probability epsilon
			if random.random() < self.epsilon:
				action = random.randint(0, 63)
			else:
				# Select move with highest predicted Q-value
				prediction = self.training_model.predict(state)
				action = np.argmax(prediction)
			# Convert action to (row, column) format
			self.last_action = (action // 8, action % 8)
		else:
			self.last_action = -1, -1
		return self.last_action

	def learn(self, new_state, done):
		"""
		If training is activated, the agent stores its experience and learns based on its replay memory
		:param new_state: New board state after an action
		:param done: Boolean value whether state is a final state that ends the game
		:return: None
		"""
		if self.train:
			# Store transition
			self.remember(new_state, done)
			# Mini-batch gradient descent
			self.replay()
			# Reset target network every c steps
			if self.total_moves > 0 and self.total_moves % self.c == 0:
				self.reset_target_network()

	def remember(self, new_state, done):
		"""
		Stores the transition in the replay memory. At least one state-action-state transition is required
		:param new_state: Array with currently observed board state
		:param done: Boolean value whether game is over or not
		:return: None
		"""
		if not ((self.last_state is None) or (self.last_action is None) or (self.reward is None)):
			self.memory.append([self.last_state, self.last_action, self.reward, new_state, done])

	def replay(self):
		"""
		Samples a random mini-batch from the replay memory and trains the network parameters based on it
		:return: None
		"""
		batch_size = 32
		if len(self.memory) < batch_size:
			return
		samples = random.sample(self.memory, batch_size)
		for sample in samples:
			state, action, reward, new_state, done = sample
			# Actions are given as field indexes (row, column) and must be flattened to fit the format of the prediction
			action_index = action[0] * 8 + action[1]
			target = self.target_model.predict(state)
			if done:
				target[0][action_index] = reward
			else:
				future_return = np.max(self.target_model.predict(new_state))
				target[0][action_index] = reward + self.gamma * future_return
			self.training_model.fit(state, target, epochs=1, verbose=0)

	def reset_target_network(self):
		"""
		Copies the network parameters of the training network to the target network
		:return: None
		"""
		self.training_model.save_weights('training_results/weights_{}.h5'.format(self.colour), overwrite=True)
		self.target_model.load_weights('training_results/weights_{}.h5'.format(self.colour))
		self.target_model.compile(loss='mse', optimizer='sgd')

	def store_network(self):
		"""
		Stores the network parameters of the training model
		:return: None
		"""
		self.training_model.save_weights('training_results/final_weights_{}.h5'.format(self.colour), overwrite=True)


class RandomAgent(Player):
	"""
	Agent making random moves
	"""

	def __init__(self, colour):
		Player.__init__(self, colour)
		self.moves = 0

	def select_action(self, state):
		"""
		Selects a random action out of all given valid actions
		:param state: Current board state
		:return: Selected move as a tuple for the position of the field; (-1,-1) if no valid move is available
		"""
		self.moves += 1
		if len(self.valid_moves) > 0:
			return tuple(self.valid_moves[np.random.choice(len(self.valid_moves))])
		else:
			return -1, -1
