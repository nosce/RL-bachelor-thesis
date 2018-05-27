import pygame
import sys
import numpy as np
from pygame.locals import *

np.random.seed(42)
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
# Rewards
GAME_WON = 1
GAME_LOST = -1
GAME_DRAW = 0
VALID_MOVE = 0
INVALID_MOVE = -100


class OthelloGame(object):
	def __init__(self):
		self.player1 = Player('black')
		self.player2 = Player('white')
		self.play_game(self.player1, self.player2)

	@staticmethod
	def play_game(player_b, player_w):
		while True:
			# Set up screen with restart button
			screen.fill(WHITE)
			font = pygame.font.SysFont("Arial", 32)
			restart = pygame.draw.rect(screen, WHITE, (100, 700, WINDOW[0] / 2, WINDOW[0] / 5))
			screen.blit(font.render("Restart", True, BLACK), (275, 700))
			# Initialize board
			game_board = Board()
			players = {player_b.id: player_b.colour, player_w.id: player_w.colour}
			current_player = player_b
			play_again = False
			# Timer
			FPS = 20
			fps_clock = pygame.time.Clock()
			# Game loop
			while game_board.game_running:
				fps_clock.tick(FPS)
				# Check if game is still running
				if play_again:
					break
				if game_board.game_running and game_board.gameover():
					if game_board.winner == 0:
						message = font.render("No winner", True, RED)
					else:
						message = font.render("{} wins".format(players[game_board.winner]), True, RED)
					screen.blit(message, (100, 640))

				for event in pygame.event.get():
					all_valid_moves = game_board.get_valid_moves(current_player)
					# Switch player immediately if no valid move possible
					if len(all_valid_moves) == 0:
						current_player = player_w if current_player == player_b else player_b
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
							valid_move = game_board.update_board(move, current_player)
							# Switch current player
							if valid_move and not game_board.gameover():
								current_player = player_w if current_player == player_b else player_b
					# Redisplay
					pygame.display.flip()


class Board(object):
	# Represents the Tic Tac Toe board
	def __init__(self):
		self.board = np.zeros((8, 8), dtype=int)
		self.fields = {}
		self.winner = False
		self.game_running = True
		self.valid_moves = []
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
			player = 1 if field_x == field_y else -1
			self.board[field_x, field_y] = player
			self.fields[field_x, field_y].draw_mark(player)

	def get_clicked_field(self, mouse_x, mouse_y):
		# Returns the id of the clicked field
		for field_id, field in self.fields.items():
			if field.is_clicked(mouse_x, mouse_y):
				return field_id
		else:
			return -1, -1

	def get_valid_moves(self, player):
		# Checks whether there are valid moves for the player
		self.clear_valid_moves(self.valid_moves)
		empty_fields = np.transpose(np.where(self.board == 0))
		for field in empty_fields:
			if self.find_flanks(field, player.id, False):
				self.valid_moves.append(field.tolist())
		self.draw_valid_moves(self.valid_moves)
		return self.valid_moves

	def draw_valid_moves(self, moves):
		# Highlights the fields where moves are possible
		if moves:
			for move_x, move_y in moves:
				self.fields[move_x, move_y].highlight()

	def clear_valid_moves(self, moves):
		# Resets the highlighted fields for possible moves
		if moves:
			for move_x, move_y in moves:
				if self.board[move_x, move_y] == 0:
					self.fields[move_x, move_y].reset()
			self.valid_moves.clear()

	def update_board(self, field_id, player):
		# Updates the board when a field has been selected
		valid_move = self.find_flanks(field_id, player.id, True)
		if field_id == (-1, -1):
			pass
		elif valid_move:
			self.board[field_id] = player.id
			field = self.fields[field_id]
			field.draw_mark(player.id)
		return valid_move

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
		return 0 <= x <= 7 and 0 <= y <= 7

	def gameover(self):
		# TODO Game over condition: board full or both players no valid move
		# if pos_row_sum == 3 or pos_column_sum == 3 or diagonal_sum == 3 or antidiagonal_sum == 3:
		# 	self.winner = 1
		# 	self.game_running = False
		# elif neg_row_sum == -3 or neg_column_sum == -3 or diagonal_sum == -3 or antidiagonal_sum == -3:
		# 	self.winner = -1
		# 	self.game_running = False
		# elif np.prod(self.board) != 0:
		# 	self.winner = 0
		# 	self.game_running = False
		return not self.game_running


class Field(object):
	# Represents a field on the board
	def __init__(self, pos_x, pos_y):
		self.x = pos_x
		self.y = pos_y
		self.rect = pygame.draw.rect(screen, WHITE, (self.x, self.y, 65, 65))

	def highlight(self):
		self.rect = pygame.draw.rect(screen, GREEN, (self.x, self.y, 65, 65))

	def reset(self):
		self.rect = pygame.draw.rect(screen, WHITE, (self.x, self.y, 65, 65))

	def is_clicked(self, mouse_x, mouse_y):
		# Checks if the field has been clicked
		return True if self.rect.collidepoint(mouse_x, mouse_y) else False

	def draw_mark(self, player_id):
		# Draws the mark of the current player on the field
		if player_id == 1:
			self.rect = pygame.draw.rect(screen, WHITE, (self.x, self.y, 65, 65))
			pygame.draw.circle(screen, BLACK, (self.x + 32, self.y + 32), 28, 0)
		else:
			self.rect = pygame.draw.rect(screen, WHITE, (self.x, self.y, 65, 65))
			pygame.draw.circle(screen, GRAY, (self.x + 32, self.y + 32), 28, 0)
			pygame.draw.circle(screen, DARKGRAY, (self.x + 32, self.y + 32), 28, 1)


class Player(object):
	# Represents the players X and O
	def __init__(self, colour):
		self.id = 1 if colour == 'black' else -1
		self.colour = colour


# Start application
if __name__ == '__main__':
	# Start playing game
	OthelloGame()
