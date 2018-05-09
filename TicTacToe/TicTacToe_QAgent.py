import pygame
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from time import process_time, perf_counter

random.seed(42)  # for reproducibility
# GUI configuration
GRAY = (245, 245, 245)
BLACK = (41, 40, 48)
YELLOW = (229, 207, 74)
BLUE = (56, 113, 193)
WINDOW = (400, 550)
# PyGame initialization
pygame.init()
screen = pygame.display.set_mode(WINDOW)
pygame.display.set_caption('Tic Tac Toe')
# Rewards
GAME_WON = 1
GAME_LOST = -1
GAME_DRAW = 0
VALID_MOVE = 0
INVALID_MOVE = -100


class TicTacToeGame(object):
	def __init__(self, learning_rate=0.5, epsilon=0.8, episodes=500, write_statistics=True):
		self.statistic = Statistics()
		self.write_statistics = write_statistics
		# Agents and learning variables
		self.learning_rate = learning_rate
		self.learning_rate_decay = np.linspace(0, learning_rate, num=episodes)
		self.epsilon = epsilon
		self.epsilon_decay = np.linspace(0, epsilon - 0.1, num=episodes)
		self.player1 = QAgent('X', alpha=self.learning_rate, gamma=0.9, epsilon=self.epsilon)
		self.player2 = QAgent('O', alpha=self.learning_rate, gamma=0.9, epsilon=self.epsilon)
		self.start_playing_episodes(episodes)

	def start_playing_episodes(self, episodes):
		# Start measuring time
		clock_start = perf_counter()
		cpu_start = process_time()
		# Start playing episodes
		for episode in range(episodes):
			result, winner = self.play_game(self.player1, self.player2)
			self.statistic.store_statistics(episode, result, winner)
			new_alpha = self.learning_rate - self.learning_rate_decay[episode]
			new_epsilon = self.epsilon - self.epsilon_decay[episode]
			self.player1.adjust_learning_values(new_alpha, new_epsilon)
			self.player2.adjust_learning_values(new_alpha, new_epsilon)
		# Write statistics
		cpu_end = process_time()
		clock_end = perf_counter()
		cpu_time = cpu_end - cpu_start
		clock_time = clock_end - clock_start
		if self.write_statistics:
			self.statistic.write_statistic_file(self.player1, episodes, clock_time, cpu_time)
			self.statistic.write_qtable(self.player1, episodes)
			self.statistic.draw_cumulative_reward()
		print("Training finished")

	@staticmethod
	def play_game(player_x, player_o):
		# Initialize board and players
		game_board = Board()
		players = {player_x.id: player_x, player_o.id: player_o}
		player_x.start_game()
		player_o.start_game()
		current_player = player_x
		final_result = {}
		fps_clock = pygame.time.Clock()
		# Game loop
		while game_board.game_running:
			# Timer; when a framerate is provided as argument, the game can be slowed down
			fps_clock.tick()
			# Give rewards when game is won / lost / drawn
			if game_board.gameover():
				for player_id, player in players.items():
					if game_board.winner == 0:
						reward = GAME_DRAW
					elif player_id == game_board.winner:
						reward = GAME_WON
					else:
						reward = GAME_LOST
					player.store_reward(reward)
					state = game_board.get_board_state()
					player.update_qtable(state)
			# Moves of agent
			else:
				# Get current board state
				state = game_board.get_board_state()
				current_player.update_qtable(state)
				# Select action
				action = current_player.select_action(state)
				valid_move, reward = game_board.update_board(action, current_player)
				current_player.store_reward(reward)
				# Switch player
				current_player = player_o if current_player == player_x else player_x
			# Redisplay board
			pygame.display.flip()
		# At the end of game return the players' rewards
		for player_id, player in players.items():
			final_result[player.mark] = player.total_reward
		return final_result, game_board.winner


class Board(object):
	# Represents the Tic Tac Toe board
	def __init__(self):
		self.board = np.zeros((3, 3), dtype=int)
		self.fields = {}
		self.winner = False
		self.game_running = True
		self.draw_board()

	def draw_board(self):
		# Draws the initial board
		screen.fill(GRAY)
		pygame.draw.rect(screen, BLACK, (40, 40, 320, 320))
		for row_index, row in enumerate(self.board):
			for col_index, column in enumerate(row):
				x_pos = col_index * 110 + 40
				y_pos = row_index * 110 + 40
				self.fields[(row_index, col_index)] = Field(x_pos, y_pos)

	def update_board(self, field_id, player):
		# Updates the board when a field has been selected
		valid_move = False
		reward = INVALID_MOVE
		if self.board[field_id] == 0:
			valid_move = True
			reward = VALID_MOVE
			self.board[field_id] = player.id
			field = self.fields[field_id]
			field.draw_mark(player.id)
		return valid_move, reward

	def get_board_state(self):
		# An array cannot be a dictionary key, the board state is therefore returned as tuple
		return tuple(map(tuple, self.board))

	def gameover(self):
		# Checks if a row, column or diagonal has a winning constellation and determines the winner
		pos_row_sum = np.max(np.sum(self.board, axis=1))
		pos_column_sum = np.max((np.sum(self.board, axis=0)))
		neg_row_sum = np.min(np.sum(self.board, axis=1))
		neg_column_sum = np.min((np.sum(self.board, axis=0)))
		diagonal_sum = np.trace(self.board)
		antidiagonal_sum = np.trace(np.fliplr(self.board))
		if pos_row_sum == 3 or pos_column_sum == 3 or diagonal_sum == 3 or antidiagonal_sum == 3:
			self.winner = 1
			self.game_running = False
		elif neg_row_sum == -3 or neg_column_sum == -3 or diagonal_sum == -3 or antidiagonal_sum == -3:
			self.winner = -1
			self.game_running = False
		elif np.prod(self.board) != 0:
			self.winner = 0
			self.game_running = False
		return not self.game_running


class Field(object):
	# Represents a field on the board
	def __init__(self, pos_x, pos_y):
		self.x = pos_x
		self.y = pos_y
		self.rect = pygame.draw.rect(screen, GRAY, (self.x, self.y, 100, 100))

	def draw_mark(self, player_id):
		# Draws the mark of the current player on the field
		if player_id == 1:
			pygame.draw.line(screen, YELLOW, (self.x + 20, self.y + 20), (self.x + 80, self.y + 80), 10)
			pygame.draw.line(screen, YELLOW, (self.x + 80, self.y + 20), (self.x + 20, self.y + 80), 10)
		else:
			pygame.draw.circle(screen, BLUE, (self.x + 50, self.y + 50), 40, 5)


class Player(object):
	# Represents the players X and O
	def __init__(self, mark):
		self.id = 1 if mark == 'X' else -1
		self.mark = mark


class QAgent(Player):
	def __init__(self, mark, epsilon, alpha, gamma):
		Player.__init__(self, mark)
		self.epsilon = epsilon  # chance of exploration instead of exploitation
		self.alpha = alpha  # learning rate
		self.gamma = gamma  # discount factor for future rewards
		self.qtable = {}  # Q-table for storing state-action-values
		self.last_state = None
		self.last_action = None
		self.reward = None
		self.total_reward = 0  # Reward accumulated during the game

	def start_game(self):
		# Reset memory of last states, action and rewards when new game starts
		self.last_state = None
		self.last_action = None
		self.reward = None
		self.total_reward = 0

	def adjust_learning_values(self, new_alpha, new_epsilon):
		# Adjust learning rate and exploration over course of episodes
		self.alpha = new_alpha
		self.epsilon = new_epsilon

	@staticmethod
	def possible_actions(state):
		# Return all empty fields in the given state
		actions = []
		for row in range(3):
			for col in range(3):
				if state[row][col] == 0:
					actions.append((row, col))
		return actions

	def select_action(self, board_state):
		# Selects an action epsilon-greedily
		# Store board state
		self.last_state = board_state
		# Randomly choose a number between 0 and 1; if it is lower than epsilon, explore possible actions randomly
		if np.random.random() < self.epsilon:
			actions = self.possible_actions(board_state)
			self.last_action = random.choice(actions)
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
				move = random.choice(all_qvalues[max_qvalue])
			else:
				move = all_qvalues[max_qvalue][0]
			self.last_action = move
		return self.last_action

	def store_reward(self, reward):
		# Remember the reward given and add it to total reward
		self.reward = reward
		self.total_reward += reward

	def update_qtable(self, new_state):
		# Recalculate Q-values if values are available (i.e. after one state-action-state transition)
		if self.last_state and self.last_action and self.reward:
			self.learn_qvalue(self.last_state, self.last_action, self.reward, new_state)

	def print_qtable(self):
		# Return Q-table in json-format, e.g to store it in a file
		json_table = {}
		for key, value in self.qtable.items():
			json_table[str(key)] = value
		return json.dumps(json_table, indent=3, sort_keys=True)

	def get_qvalue(self, state, action):
		# Return Q-value for state-action pair if it exists, otherwise start with a Q-value of 0.1
		# to encourage exploration of new states
		if (state, action) not in self.qtable:
			self.qtable[(state, action)] = 0.1
		return self.qtable[(state, action)]

	def learn_qvalue(self, state, action, reward, new_state):
		# Update Q-values based on action-value-function Q(s,a)
		current_value = self.get_qvalue(state, action)
		possible_returns = [self.get_qvalue(new_state, a) for a in self.possible_actions(new_state)]
		max_return = max(possible_returns) if possible_returns else 0
		self.qtable[(state, action)] = current_value + self.alpha * (reward + self.gamma * max_return - current_value)


class Statistics(object):
	# Bundles functions for getting learning statistics
	def __init__(self):
		self.episodes = []
		self.win_x = 0
		self.win_o = 0
		self.draw = 0
		self.rewards_x = []
		self.rewards_o = []
		self.cumulative_reward_x = 0
		self.cumulative_reward_o = 0
		self.cum_reward_list_x = []
		self.cum_reward_list_o = []

	def store_statistics(self, episode, result, winner):
		self.count_winner(winner)
		self.episodes.append(episode)
		self.rewards_x.append(result["X"])
		self.rewards_o.append(result["O"])

	def count_winner(self, player_id):
		if player_id == 1:
			self.win_x += 1
		elif player_id == -1:
			self.win_o += 1
		elif player_id == 0:
			self.draw += 1

	def calc_rewards(self):
		for x, o in zip(self.rewards_x, self.rewards_o):
			self.cumulative_reward_x += x
			self.cum_reward_list_x.append(self.cumulative_reward_x)
			self.cumulative_reward_o += o
			self.cum_reward_list_o.append(self.cumulative_reward_o)

	def draw_cumulative_reward(self):
		self.calc_rewards()
		plt.plot(self.episodes, self.cum_reward_list_x, label='Player X', color='orange')
		plt.plot(self.episodes, self.cum_reward_list_o, label='Player O', color='blue')
		plt.ylabel('Cumulative reward')
		plt.xlabel('Episode')
		plt.legend()
		plt.show()

	def write_statistic_file(self, player, eps, clock_time, cpu_time):
		filename = "trainingresults_Q_{}.txt".format(eps)
		file = open(filename, "w+")
		file.write("Elapsed time: {} sec".format(clock_time))
		file.write("\nCPU processing time: {} sec".format(cpu_time))
		file.write("\nNumber of entries in Q-Table of player {}: {}".format(player.mark, len(player.qtable)))
		file.write("\nPlayer X won: {} \nPlayer O won: {}\nDraw: {}".format(self.win_x, self.win_o, self.draw))
		file.write("\nAverage reward of Player X: {}".format(np.mean(self.rewards_x)))
		file.write("\nAverage reward of Player O: {}".format(np.mean(self.rewards_o)))
		print("Stored results after {} episodes".format(eps))

	def write_qtable(self, player, eps):
		filename = "qtable{}.json".format(eps)
		file = open(filename, "w+")
		file.write(player.print_qtable())
		print("Saved Q-table for player {}".format(player.mark))


# Start application
if __name__ == '__main__':
	# Start playing episodes of game with learning and exploration rate and writing statistics
	TicTacToeGame(0.5, 0.8, 5000, True)
