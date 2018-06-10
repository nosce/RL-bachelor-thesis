"""
This file contains the game logic of Tic Tac Toe. The board is represented as a 3 x 3 matrix. Fields that player X
has marked are stored as "1", marked fields of player O are stored as "-1", empty fields are "0".
Each field of the board has an id given as (row_number, column_number).
Players collect rewards while playing the game: 1 if they win, -1 if they lose and 0 if the game is a draw.
"""
import pygame
import numpy as np

__author__ = "Claudia Kutter"
__license__ = "GPL"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

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
	"""
	Implements the game logic how players act on the board
	"""

	def __init__(self, player_x, player_o):
		self.game_board = Board()
		self.player_x = player_x
		self.player_o = player_o

	def play_game(self, epsilon, alpha):
		"""
		Main game loop: Plays Tic Tac Toe until game is over
		:param epsilon: Exploration rate which the agents are to use during the game
		:param alpha: Learning rate which the agents are to use during the game
		:return: Dictionary of game statistics
		"""
		self.game_board.reset_for_new_game()
		self.player_x.reset_for_new_game(epsilon=epsilon, alpha=alpha)
		self.player_o.reset_for_new_game(epsilon=epsilon, alpha=alpha)
		players = [self.player_x, self.player_o]
		current_player = self.player_x
		final_result = {}
		# Timer
		fps_clock = pygame.time.Clock()
		# Game loop
		while self.game_board.game_running:
			fps_clock.tick()
			# Give rewards when game is won / lost / drawn
			if self.game_board.gameover():
				state = self.game_board.get_board_state()
				for player in players:
					if self.game_board.winner == 0:
						reward = GAME_DRAW
					elif player.id == self.game_board.winner:
						reward = GAME_WON
					else:
						reward = GAME_LOST
					player.store_reward(reward)
					player.learn(state)
				break

			# Moves of agent
			else:
				# Get current board state
				state = self.game_board.get_board_state()
				action = current_player.learn(state)
				valid_move, reward = self.game_board.update_board(action, current_player)
				current_player.store_reward(reward)
				# Switch player
				current_player = self.player_o if current_player == self.player_x else self.player_x

			# Redisplay board
			pygame.display.flip()

		# At the end of game return the final results of the game
		for player in players:
			final_result[player.mark] = {"reward": player.total_reward,
										 "moves": player.moves}
		final_result["winner"] = self.game_board.winner
		final_result["explore"] = epsilon
		return final_result


class Board(object):
	"""
	Represents the Tic Tac Toe board as a 3 x 3 matrix and handles board updates and move calculations
	"""

	def __init__(self):
		self.board = np.zeros((3, 3), dtype=int)
		self.fields = {}
		self.winner = False
		self.game_running = True
		self.draw_board()

	def reset_for_new_game(self):
		"""
		Resets the board to the initial state for a new game episode
		:return: False
		"""
		self.__init__()

	def draw_board(self):
		"""
		Draws the board
		:return: None
		"""
		screen.fill(GRAY)
		pygame.draw.rect(screen, BLACK, (40, 40, 320, 320))
		for row_index, row in enumerate(self.board):
			for col_index, column in enumerate(row):
				x_pos = col_index * 110 + 40
				y_pos = row_index * 110 + 40
				self.fields[(row_index, col_index)] = Field(x_pos, y_pos)

	def update_board(self, field_id, player):
		"""
		Updates the board state when a player makes a move
		:param field_id: Field (as tuple) where a player places his mark
		:param player: Id of the player making the move
		:return: Boolean value whether it is a valid move and reward for the move
		"""
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
		"""
		Returns the board state as immutable tuple so that it can be used as key in a dictionary
		:return: Board state as a tuple
		"""
		return tuple(map(tuple, self.board))

	def gameover(self):
		"""
		Checks whether the game is over because the board is full or a player has three in a row
		:return: Boolean value whether game is still running (True) or not (False)
		"""
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
	"""
	Represents a field on the board graphically
	"""

	def __init__(self, pos_x, pos_y):
		self.x = pos_x
		self.y = pos_y
		self.rect = pygame.draw.rect(screen, GRAY, (self.x, self.y, 100, 100))

	def draw_mark(self, player_id):
		"""
		Draws the player's mark on the field
		:param player_id: ID of the player; 1 means X, -1 means O
		:return: None
		"""
		if player_id == 1:
			pygame.draw.line(screen, YELLOW, (self.x + 20, self.y + 20), (self.x + 80, self.y + 80), 10)
			pygame.draw.line(screen, YELLOW, (self.x + 80, self.y + 20), (self.x + 20, self.y + 80), 10)
		else:
			pygame.draw.circle(screen, BLUE, (self.x + 50, self.y + 50), 40, 5)
