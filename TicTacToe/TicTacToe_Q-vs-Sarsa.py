import pygame
import numpy as np
import json
import matplotlib.pyplot as plt
from time import process_time, perf_counter

np.random.seed(42)  # for reproducibility
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
		self.epsilon_decay = np.linspace(0, epsilon, num=episodes)
		self.player1 = QAgent('X', alpha=self.learning_rate, gamma=0.9, epsilon=self.epsilon)
		self.player2 = SarsaAgent('O', alpha=self.learning_rate, gamma=0.9, epsilon=self.epsilon)
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
			self.statistic.write_statistic_file(self.player1, self.player2, episodes, clock_time, cpu_time)
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
			final_result[player.mark] = {"reward": player.total_reward, "moves": player.moves}
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


class Agent(Player):
	def __init__(self, mark, epsilon, alpha, gamma):
		Player.__init__(self, mark)
		self.epsilon = epsilon  # chance of exploration instead of exploitation
		self.alpha = alpha  # learning rate
		self.gamma = gamma  # discount factor for future rewards
		self.qtable = {}  # Q-table for storing state-action-values
		self.total_reward = 0  # Reward accumulated during the game
		self.last_state = None
		self.last_action = None
		self.reward = None
		self.moves = 0

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

	def store_reward(self, reward):
		# Remember the reward given and add it to total reward
		self.reward = reward
		self.total_reward += reward

	def get_qvalue(self, state, action):
		# Return Q-value for state-action pair if it exists, otherwise start with a Q-value of 0
		# in order to encourage exploration of new states
		if (state, action) not in self.qtable:
			self.qtable[(state, action)] = 0.1
		return self.qtable[(state, action)]


class QAgent(Agent):
	def __init__(self, mark, epsilon, alpha, gamma):
		Agent.__init__(self, mark, epsilon, alpha, gamma)
		self.method = "Q-Learning"

	def start_game(self):
		# Reset memory of last states, action and rewards when new game starts
		self.last_state = None
		self.last_action = None
		self.reward = None
		self.total_reward = 0
		self.moves = 0

	def select_action(self, board_state):
		# Selects an action epsilon-greedily
		# Store board state
		self.last_state = board_state
		# Randomly choose a number between 0 and 1; if it is lower than epsilon, explore possible actions randomly
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
		# Update Q-values based on action-value-function Q(s,a) if values are available
		# (i.e. after one state-action-state transition)
		if self.last_state and self.last_action and not (self.reward is None):
			current_value = self.get_qvalue(self.last_state, self.last_action)
			possible_returns = [self.get_qvalue(new_state, a) for a in self.possible_actions(new_state)]
			max_return = max(possible_returns) if possible_returns else 0
			self.qtable[(self.last_state, self.last_action)] = current_value + self.alpha * (self.reward + self.gamma *
																							 max_return - current_value)

	def print_qtable(self):
		# Return Q-table in json-format, e.g to store it in a file
		json_table = {}
		for key, value in self.qtable.items():
			json_table[str(key)] = value
		return json.dumps(json_table, indent=3, sort_keys=True)


class SarsaAgent(Agent):
	def __init__(self, mark, epsilon, alpha, gamma):
		Agent.__init__(self, mark, epsilon, alpha, gamma)
		self.method = "SARSA"
		self.current_state = None
		self.current_action = None

	def start_game(self):
		# Reset memory of last states, action and rewards when new game starts
		self.current_state = None
		self.current_action = None
		self.last_state = None
		self.last_action = None
		self.reward = None
		self.total_reward = 0
		self.moves = 0

	def select_action(self, board_state):
		# Selects an action epsilon-greedily
		# Store previous values
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

	def update_qtable(self, new_state):
		# Recalculate Q-values if values are available (i.e. after one state-action-state transition)
		if self.last_state and self.last_action and not (self.reward is None):
			current_value = self.get_qvalue(self.last_state, self.last_action)
			next_return = self.get_qvalue(self.current_state, self.current_action)
			self.qtable[(self.last_state, self.last_action)] = current_value + self.alpha * \
															   (self.reward + self.gamma * next_return - current_value)

	def print_qtable(self):
		# Return Q-table in json-format, e.g to store it in a file
		json_table = {}
		for key, value in self.qtable.items():
			json_table[str(key)] = value
		return json.dumps(json_table, indent=3, sort_keys=True)


class Statistics(object):
	# Bundles functions for getting learning statistics
	def __init__(self):
		self.episodes = []
		self.wins = {"X": 0, "O": 0, "D": 0}
		self.rewards = {"X": [], "O": []}
		self.moves = {"X": [], "O": []}
		self.cumulative_reward = {"X": 0, "O": 0}
		self.cum_reward_list = {"X": [], "O": []}

	def store_statistics(self, episode, result, winner):
		self.count_winner(winner)
		self.episodes.append(episode)
		self.rewards["X"].append(result["X"]["reward"])
		self.rewards["O"].append(result["O"]["reward"])
		self.moves["X"].append(result["X"]["moves"])
		self.moves["O"].append(result["O"]["moves"])
		self.cumulative_reward["X"] += result["X"]["reward"]
		self.cum_reward_list["X"].append(self.cumulative_reward["X"])
		self.cumulative_reward["O"] += result["O"]["reward"]
		self.cum_reward_list["O"].append(self.cumulative_reward["O"])

	def count_winner(self, player_id):
		if player_id == 1:
			self.wins["X"] += 1
		elif player_id == -1:
			self.wins["O"] += 1
		elif player_id == 0:
			self.wins["D"] += 1

	def draw_cumulative_reward(self):
		plt.plot(self.episodes, self.cum_reward_list["X"], label='Player X', color='orange')
		plt.plot(self.episodes, self.cum_reward_list["O"], label='Player O', color='blue')
		plt.ylabel('Cumulative reward')
		plt.xlabel('Episode')
		plt.legend()
		plt.show()

	def write_statistic_file(self, player_x, player_o, eps, clock_time, cpu_time):
		filename = "trainingresults_{}-vs-{}_{}.txt".format(player_x.method, player_o.method, eps)
		file = open(filename, "w+")
		file.write("Elapsed time: {} sec".format(clock_time))
		file.write("\nCPU processing time: {} sec".format(cpu_time))
		file.write("\n" + ("*" * 40))
		file.write("\nPlayer X trained with {}".format(player_x.method))
		file.write("\nNumber of entries in Q-Table of player X: {}".format(len(player_x.qtable)))
		file.write("\nPlayer X won: {}".format(self.wins["X"]))
		file.write("\nAverage reward of Player X: {}".format(np.mean(self.rewards["X"])))
		file.write("\nTotal reward of Player X after {} games: {}".format(eps, self.cumulative_reward["X"]))
		file.write("\nAverage number of moves of Player X: {}".format(np.mean(self.moves["X"])))
		file.write("\n" + ("*" * 40))
		file.write("\nPlayer O trained with {}".format(player_o.method))
		file.write("\nNumber of entries in Q-Table of player O: {}".format(len(player_o.qtable)))
		file.write("\nPlayer O won: {}".format(self.wins["O"]))
		file.write("\nAverage reward of Player O: {}".format(np.mean(self.rewards["O"])))
		file.write("\nTotal reward of Player O after {} games: {}".format(eps, self.cumulative_reward["O"]))
		file.write("\nAverage number of moves of Player O: {}".format(np.mean(self.moves["O"])))
		file.write("\nDraw: {}".format(self.wins["D"]))
		print("Stored results after {} episodes".format(eps))

	@staticmethod
	def write_qtable(player, eps):
		filename = "qtable{}_{}.json".format(eps, player.method)
		file = open(filename, "w+")
		file.write(player.print_qtable())
		print("Saved Q-table for player {}".format(player.mark))


# Start application
if __name__ == '__main__':
	# Start playing episodes of game with learning and exploration rate and writing statistics
	TicTacToeGame(0.5, 0.8, 160000, True)
