import pygame
import sys
import numpy as np
from pygame.locals import *

# GUI configuration
WHITE = (255, 255, 255)
BLACK = (41, 40, 48)
GRAY = (246, 246, 246)
DARKGRAY = (195, 195, 195)
RED = (205, 35, 84)
GREEN = (233, 243, 237)
WINDOW = (620, 800)
# PyGame initialization
pygame.init()
screen = pygame.display.set_mode(WINDOW)
pygame.display.set_caption('Othello')


class OthelloGame(object):
	def __init__(self):
		self.player_b = Player('black')
		self.player_w = Player('white')

	def play_game(self):
		while True:
			# Set up screen with restart button
			screen.fill(WHITE)
			font = pygame.font.SysFont("Arial", 32)
			restart = pygame.draw.rect(screen, WHITE, (100, 720, WINDOW[0] / 2, WINDOW[0] / 5))
			screen.blit(font.render("Restart", True, BLACK), (275, 730))
			# Initialize board
			game_board = Board()
			players = {self.player_b.id: self.player_b.colour, self.player_w.id: self.player_w.colour}
			current_player = self.player_b
			play_again = False
			black = 2
			white = 2
			# Timer
			fps = 20
			fps_clock = pygame.time.Clock()
			# Game loop
			while not play_again:
				fps_clock.tick(fps)
				# Check if game is still running
				if play_again:
					break
				if game_board.game_running and game_board.gameover():
					if game_board.winner == 0:
						message = font.render("No winner", True, RED)
					else:
						message = font.render("{} wins".format(players[game_board.winner]), True, RED)
					screen.blit(message, (260, 680))
				else:
					# Check whether there are possible moves, else switch player
					game_board.clear_highlights()
					all_valid_moves = current_player.get_valid_moves(game_board)
					if len(all_valid_moves) != 0:
						game_board.draw_valid_moves(all_valid_moves)
					else:
						current_player = self.player_w if current_player == self.player_b else self.player_b

				for event in pygame.event.get():
					# Wait for input
					if event.type == QUIT or (event.type == KEYUP and event.key == K_ESCAPE):
						pygame.quit()
						sys.exit()
					elif event.type == MOUSEBUTTONDOWN and event.button == 1:
						mouse_x, mouse_y = event.pos
						# Restart game if restart button is clicked
						play_again = restart.collidepoint(mouse_x, mouse_y)
						# Check for board input if the game is running
						if not game_board.gameover():
							move = game_board.get_clicked_field(mouse_x, mouse_y)
							valid_move, black, white = game_board.update_board(move, current_player)
							# Switch current player after a valid move
							if valid_move and not game_board.gameover():
								current_player = self.player_w if current_player == self.player_b else self.player_b
					# Redisplay
					pygame.draw.rect(screen, WHITE, (75, 620, WINDOW[0], 35))
					message = font.render("Black stones: {}  |  White stones: {}".format(black, white), True, DARKGRAY)
					screen.blit(message, (75, 620))
					pygame.display.flip()


class Board(object):
	# Represents the Othello board
	def __init__(self):
		self.board = np.zeros((8, 8), dtype=int)
		self.fields = {}
		self.no_moves_possible = {}
		self.black = 2
		self.white = 2
		self.game_running = True
		self.winner = False
		self.draw_board()

	def draw_board(self):
		# Draws the initial board
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

	def get_clicked_field(self, mouse_x, mouse_y):
		# Returns the id of the clicked field
		for field_id, field in self.fields.items():
			if field.is_clicked(mouse_x, mouse_y):
				return field_id
		else:
			return -1, -1

	def draw_valid_moves(self, moves):
		# Highlights the fields where moves are possible
		if moves:
			for move_x, move_y in moves:
				self.fields[move_x, move_y].highlight()

	def clear_highlights(self):
		# Resets the highlighted fields for possible moves
		empty_fields = np.transpose(np.where(self.board == 0))
		for field_x, field_y in empty_fields:
			self.fields[field_x, field_y].reset()

	def update_board(self, field_id, player):
		# Updates the board when a field has been selected
		valid_move = self.board[field_id] == 0 and self.find_flanks(field_id, player.id, True)
		if field_id == (-1, -1):
			pass
		elif valid_move:
			self.board[field_id] = player.id
			field = self.fields[field_id]
			field.draw_mark(player.id)
		# Count stones on board
		self.black = np.count_nonzero(self.board == 1)
		self.white = np.count_nonzero(self.board == -1)
		return valid_move, self.black, self.white

	def find_flanks(self, move, player, make_move):
		# Finds flanked stones
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
		# Flips stones
		for field in flip_fields:
			new_sign = -(self.board[field])
			self.board[field] = new_sign
			self.fields[field].draw_mark(new_sign)

	@staticmethod
	def field_on_board(x, y):
		# Checks whether the given field is on the board
		return 0 <= x <= 7 and 0 <= y <= 7

	def gameover(self):
		# Game over if no move possibilities left for both players
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
	Represents a field on the board graphically.
	"""
	def __init__(self, pos_x, pos_y):
		self.x = pos_x
		self.y = pos_y
		self.rect = pygame.draw.rect(screen, WHITE, (self.x, self.y, 65, 65))

	def highlight(self):
		"""
		Highlights the field graphically, e.g. because a valid move is possible there
		:return: None
		"""
		self.rect = pygame.draw.rect(screen, GREEN, (self.x, self.y, 65, 65))

	def reset(self):
		"""
		Resets highlighting of the field
		:return: None
		"""
		self.rect = pygame.draw.rect(screen, WHITE, (self.x, self.y, 65, 65))

	def is_clicked(self, mouse_x, mouse_y):
		"""
		Checks whether a field has been clicked
		:param mouse_x: x-coordinate of the cursor position
		:param mouse_y: y-coordinate of the cursor position
		:return: True if a field has been clicked
		"""
		return True if self.rect.collidepoint(mouse_x, mouse_y) else False

	def draw_mark(self, player_id):
		"""
		Draws the player's stone on the field
		:param player_id: ID of the player; 1 means black, -1 means white
		:return: None
		"""
		if player_id == 1:
			self.rect = pygame.draw.rect(screen, WHITE, (self.x, self.y, 65, 65))
			pygame.draw.circle(screen, BLACK, (self.x + 32, self.y + 32), 28, 0)
		else:
			self.rect = pygame.draw.rect(screen, WHITE, (self.x, self.y, 65, 65))
			pygame.draw.circle(screen, GRAY, (self.x + 32, self.y + 32), 28, 0)
			pygame.draw.circle(screen, DARKGRAY, (self.x + 32, self.y + 32), 28, 1)


class Player(object):
	"""
	Represents a player in the game. The black player's ID is 1, the white player's ID is -1, The IDs are used to
	mark the position of the player's stones on the board.
	"""
	def __init__(self, colour):
		self.id = 1 if colour == 'black' else -1
		self.colour = colour
		self.valid_moves = []

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


# Start application
if __name__ == '__main__':
	game = OthelloGame()
	game.play_game()
