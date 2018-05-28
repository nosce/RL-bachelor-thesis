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
		self.player1 = Player('black')
		self.player2 = Player('white')
		self.play_game(self.player1, self.player2)

	@staticmethod
	def play_game(player_b, player_w):
		while True:
			# Set up screen with restart button
			screen.fill(WHITE)
			font = pygame.font.SysFont("Arial", 32)
			restart = pygame.draw.rect(screen, WHITE, (100, 720, WINDOW[0] / 2, WINDOW[0] / 5))
			screen.blit(font.render("Restart", True, BLACK), (275, 730))
			# Initialize board
			game_board = Board()
			players = {player_b.id: player_b.colour, player_w.id: player_w.colour}
			current_player = player_b
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
					# Check whether there are possible moves
					all_valid_moves = game_board.get_valid_moves(current_player)
					if len(all_valid_moves) == 0:
						current_player = player_w if current_player == player_b else player_b

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
							# Switch current player
							if valid_move and not game_board.gameover():
								current_player = player_w if current_player == player_b else player_b
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
		self.winner = False
		self.black = 2
		self.white = 2
		self.game_running = True
		self.valid_moves = []
		self.no_moves_possible = {}
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

	def get_valid_moves(self, player):
		# Checks whether there are valid moves for the player
		self.clear_valid_moves(self.valid_moves)
		empty_fields = np.transpose(np.where(self.board == 0))
		for field in empty_fields:
			if self.find_flanks(field, player.id, False):
				self.valid_moves.append(field.tolist())
		self.draw_valid_moves(self.valid_moves)
		# Log whether there is a valid move for the player
		self.no_moves_possible[player] = True if len(self.valid_moves) == 0 else False
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
	# Represents a field on the board
	def __init__(self, pos_x, pos_y):
		self.x = pos_x
		self.y = pos_y
		self.rect = pygame.draw.rect(screen, WHITE, (self.x, self.y, 65, 65))

	def highlight(self):
		# Highlights the field
		self.rect = pygame.draw.rect(screen, GREEN, (self.x, self.y, 65, 65))

	def reset(self):
		# Clears any highlights from the field
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
	# Represents the black and white players
	def __init__(self, colour):
		self.id = 1 if colour == 'black' else -1
		self.colour = colour


# Start application
if __name__ == '__main__':
	OthelloGame()
