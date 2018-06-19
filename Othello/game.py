"""
This file contains the game logic of Othello. The board is represented as a 8 x 8 matrix. Stone positions of the
black player are marked with "1", stone positions of the white player are marked with "-1", empty fields are "0".
Each field of the board has an id given as (row_number, column_number).
Players collect rewards while playing the game: 1 if they win, -1 if they lose and 0 if the game is a draw.
"""
import pygame
import random
import numpy as np

__author__ = "Claudia Kutter"
__license__ = "GPL"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

random.seed(42)  # For reproducibility
# GUI configuration
WHITE = (255, 255, 255)
BLACK = (41, 40, 48)
YELLOW = (255, 255, 222)
DARKGRAY = (195, 195, 195)
RED = (205, 35, 84)
GREEN = (233, 243, 237)
WINDOW = (620, 800)
# PyGame initialization
pygame.init()
screen = pygame.display.set_mode(WINDOW)
pygame.display.set_caption('Othello')
# Rewards
GAME_WON = 1
GAME_LOST = -1
GAME_DRAW = 0
VALID_MOVE = 0
INVALID_MOVE = -10


class OthelloGame(object):
	"""
	Implements the game logic how players act on the board
	"""

	def __init__(self, player_black, player_white):
		self.game_board = Board()
		self.player_b = player_black
		self.player_w = player_white

	def play_game(self, eps):
		"""
		Main game loop: Plays Othello until game is over
		:param eps: Exploration rate which the agents are to use during the game
		:return: Dictionary of game statistics
		"""
		# Initialize board and players
		self.game_board.reset_for_new_game()
		self.player_b.reset_for_new_game(eps)
		self.player_w.reset_for_new_game(eps)
		players = [self.player_b, self.player_w]
		current_player = self.player_b
		final_result = {}
		# Timer
		fps_clock = pygame.time.Clock()
		# Game loop
		while True:
			fps_clock.tick()
			# Check if game is still running
			if self.game_board.gameover() or self.game_board.cheated:
				state = self.game_board.get_state()
				# Punish the player that has cheated
				if self.game_board.cheated:
					current_player.learn(state, True)
				# Store game result
				else:
					for player in players:
						if self.game_board.winner == 0:
							reward = GAME_DRAW
						elif player.id == self.game_board.winner:
							reward = GAME_WON
						else:
							reward = GAME_LOST
						player.store_reward(reward)
						player.learn(state, True)
				break

			else:
				self.game_board.clear_highlights()
				# Observe state
				state = self.game_board.get_state()
				# Check whether there are possible moves, else switch player immediately
				all_valid_moves = current_player.get_valid_moves(self.game_board)
				if len(all_valid_moves) == 0:
					current_player = self.player_w if current_player == self.player_b else self.player_b
				else:
					# Learn from the move
					current_player.learn(state, False)
					self.game_board.draw_valid_moves(all_valid_moves)
					# Select action
					action = current_player.select_action(state)
					# Execute action an observe reward
					valid_move, reward = self.game_board.update_board(action, current_player)
					current_player.store_reward(reward)
					if valid_move:
						# Switch player
						current_player = self.player_w if current_player == self.player_b else self.player_b

			# Redisplay
			pygame.display.flip()

		# At the end of game return the final results of the game
		for player in players:
			final_result[player.colour] = {"reward": player.total_reward,
										   "moves": player.moves}
		final_result["winner"] = self.game_board.winner
		final_result["explore"] = eps
		return final_result


class Board(object):
	"""
	Represents the Othello board as a 8 x 8 matrix and handles board updates and move calculations
	"""
	def __init__(self):
		self.board = np.zeros((8, 8), dtype=int)
		self.fields = {}
		self.no_moves_possible = {}
		self.black = 2
		self.white = 2
		self.game_running = True
		self.cheated = False
		self.winner = -2
		self.draw_board()

	def reset_for_new_game(self):
		"""
		Resets the board to the initial state for a new game episode
		:return: False
		"""
		self.__init__()

	def get_state(self):
		"""
		Returns the board state for feeding it into the neural network. An 8 x 8 x 2 matrix is returned representing
		the state of the board for the black player (1 if black stone placed, else 0), and the white player (1 if
		white stone placed, else 0)
		:return: Matrix of current board state
		"""
		board_state = self.board.flatten()
		board_state = np.expand_dims(board_state, axis=0)
		return board_state

	def draw_board(self):
		"""
		Draws the board and the initially placed stones
		:return: None
		"""
		screen.fill(WHITE)
		pygame.draw.rect(screen, DARKGRAY, (20, 20, 575, 575))
		for row_index, row in enumerate(self.board):
			for col_index, column in enumerate(row):
				x_pos = col_index * 70 + 30
				y_pos = row_index * 70 + 30
				self.fields[(row_index, col_index)] = Field(x_pos, y_pos)
		# Initial stones
		initial_fields = [[3, 3], [3, 4], [4, 3], [4, 4]]
		for field_x, field_y in initial_fields:
			player = -1 if field_x == field_y else 1
			self.board[field_x, field_y] = player
			self.fields[field_x, field_y].draw_mark(player)

	def draw_valid_moves(self, moves):
		"""
		Highlights the fields where moves are possible
		:param moves: Array of fields where moves are possible
		:return: None
		"""
		if moves:
			for move_x, move_y in moves:
				self.fields[move_x, move_y].highlight()

	def clear_highlights(self):
		"""
		Resets the highlighted fields for possible moves
		:return: None
		"""
		empty_fields = np.transpose(np.where(self.board == 0))
		for field_x, field_y in empty_fields:
			self.fields[field_x, field_y].reset()

	def update_board(self, field_id, player):
		"""
		Updates the board state when a player makes a move
		:param field_id: Field (as tuple) where a player places his stone
		:param player: Id of the player making the move
		:return: Boolean value whether it is a valid move and reward for the move
		"""
		valid_move = self.board[field_id] == 0 and self.find_flanks(field_id, player.id, True)
		if valid_move:
			reward = VALID_MOVE
			self.board[field_id] = player.id
			field = self.fields[field_id]
			field.draw_mark(player.id)
		else:
			self.cheated = True
			reward = INVALID_MOVE
		# Count stones on board
		self.black = np.count_nonzero(self.board == 1)
		self.white = np.count_nonzero(self.board == -1)
		return valid_move, reward

	def find_flanks(self, move, player, make_move):
		"""
		Checks whether a player's move is valid because it results in flipping the opponent's stones
		:param move: Move of the player represented as a tuple for the respective board field
		:param player: ID of the player who wants to make the move
		:param make_move: Boolean value whether the move should be executed after checking for flanks
		:return:
		"""
		valid_move = False
		directions = [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]
		for dx, dy in directions:
			flip_fields = []
			x, y = move
			x += dx
			y += dy
			# Continue to other direction if board edge is reached
			if not self.field_on_board(x, y):
				continue
			while self.board[(x, y)] == -player:
				flip_fields.append((x, y))
				x += dx
				y += dy
				# No flipping if board edge or empty field is reached
				if not self.field_on_board(x, y) or self.board[(x, y)] == 0:
					flip_fields.clear()
					break
			if len(flip_fields) != 0:
				valid_move = True
				if make_move:
					self.flip_stones(flip_fields)
		return valid_move

	def flip_stones(self, flip_fields):
		"""
		Flips the opponent's stones
		:param flip_fields: Array of board fields with the opponent's stones to be flipped
		:return: None
		"""
		for field in flip_fields:
			new_sign = -(self.board[field])
			self.board[field] = new_sign
			self.fields[field].draw_mark(new_sign)

	@staticmethod
	def field_on_board(x, y):
		"""
		Checks whether a given field position is on the board
		:param x: Row index
		:param y: Column index
		:return: Boolean value whether field is on the board (True) or not
		"""
		return 0 <= x <= 7 and 0 <= y <= 7

	def gameover(self):
		"""
		Checks whether the game is over because no player has any valid move left
		:return: Boolean value whether game is still running (True) or not (False)
		"""
		if len(self.no_moves_possible) != 0 and all(v is True for v in self.no_moves_possible.values()):
			self.game_running = False
			# Determine winner
			if self.black > self.white:
				self.winner = 1
			elif self.black < self.white:
				self.winner = -1
			else:
				self.winner = 0
		return not self.game_running


class Field(object):
	"""
	Represents a field on the board graphically
	"""
	def __init__(self, pos_x, pos_y):
		self.x = pos_x
		self.y = pos_y
		self.rect = pygame.draw.rect(screen, GREEN, (self.x, self.y, 65, 65))

	def highlight(self):
		"""
		Highlights the field graphically, e.g. because a valid move is possible there
		:return: None
		"""
		self.rect = pygame.draw.rect(screen, YELLOW, (self.x, self.y, 65, 65))

	def reset(self):
		"""
		Resets highlighting of the field
		:return: None
		"""
		self.rect = pygame.draw.rect(screen, GREEN, (self.x, self.y, 65, 65))

	def draw_mark(self, player_id):
		"""
		Draws the player's stone on the field
		:param player_id: ID of the player; 1 means black, -1 means white
		:return: None
		"""
		if player_id == 1:
			self.rect = pygame.draw.rect(screen, GREEN, (self.x, self.y, 65, 65))
			pygame.draw.circle(screen, BLACK, (self.x + 32, self.y + 32), 28, 0)
		else:
			self.rect = pygame.draw.rect(screen, GREEN, (self.x, self.y, 65, 65))
			pygame.draw.circle(screen, WHITE, (self.x + 32, self.y + 32), 28, 0)
			pygame.draw.circle(screen, DARKGRAY, (self.x + 32, self.y + 32), 28, 1)
