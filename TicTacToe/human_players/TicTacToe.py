import pygame, sys
import numpy as np
from pygame.locals import *

# Color configuration
GRAY = (245, 245, 245)
BLACK = (41, 40, 48)
RED = (205, 35, 84)
YELLOW = (229, 207, 74)
BLUE = (56, 113, 193)
# Window size
WINDOW = (400, 550)


def main():
	while True:
		# Set up screen with restart button
		screen.fill(GRAY)
		font = pygame.font.SysFont("Arial", 32)
		restart = pygame.draw.rect(screen, GRAY, (100, 450, WINDOW[0] / 2, WINDOW[0] / 5))
		screen.blit(font.render("Restart", True, BLACK), (150, 470))
		# Initialize board and players
		player_x = Player('X')
		player_o = Player('O')
		game_board = Board()
		current_player = player_x
		play_again = False
		# Timer
		FPS = 20
		fpsClock = pygame.time.Clock()
		# Game loop
		while True:
			fpsClock.tick(FPS)
			# Event handling
			for event in pygame.event.get():
				if event.type == QUIT or (event.type == KEYUP and event.key == K_ESCAPE):
					pygame.quit()
					sys.exit()
				elif event.type == MOUSEBUTTONDOWN and event.button == 1:
					mouse_x, mouse_y = event.pos
					# Restart game if restart button is clicked
					play_again = restart.collidepoint(mouse_x, mouse_y)
					# Check for board input if the game is running
					if not game_board.gameover() or game_board.tie():
						valid_move = game_board.update_board(mouse_x, mouse_y, current_player)
						# Switch current player
						if valid_move and not game_board.gameover():
							current_player = player_o if current_player == player_x else player_x
				# Redisplay
				pygame.display.flip()
			if play_again:
				break
			if game_board.gameover():
				message = font.render("Player {} wins".format(current_player.mark), True, RED)
				screen.blit(message, (100, 400))
			elif game_board.tie():
				message = font.render("No winner", True, RED)
				screen.blit(message, (100, 400))


class Board(object):
	# Represents the Tic Tac Toe board
	def __init__(self):
		self.board = np.zeros((3, 3), dtype=int)
		self.fields = []
		self.draw_board()

	def draw_board(self):
		# Draws the initial board
		pygame.draw.rect(screen, BLACK, (40, 40, 320, 320))
		for row_index, row in enumerate(self.board):
			for col_index, column in enumerate(row):
				x_pos = col_index * 110 + 40
				y_pos = row_index * 110 + 40
				field = Field((row_index, col_index), x_pos, y_pos)
				self.fields.append(field)

	def update_board(self, mouse_x, mouse_y, player):
		# Updates the board when a field has been clicked
		valid_move = False
		for field in self.fields:
			if field.is_clicked(mouse_x, mouse_y) and self.board[field.id] == 0:
				valid_move = True
				self.board[field.id] = player.id
				field.draw_mark(player.id)
				break
		return valid_move

	def gameover(self):
		# Checks if a row, column or diagonal has a winning constellation
		row_sum = np.max(np.absolute(np.sum(self.board, axis=1)))
		column_sum = np.max(np.absolute(np.sum(self.board, axis=0)))
		diagonal_sum = np.absolute(np.trace(self.board))
		antidiagonal_sum = np.absolute(np.trace(np.fliplr(self.board)))
		return True if row_sum == 3 or column_sum == 3 or diagonal_sum == 3 or antidiagonal_sum == 3 else False

	def tie(self):
		# Checks if all fields are marked but there is no winning constellation
		return False if np.prod(self.board) == 0 else True


class Field(object):
	# Represents a field on the board
	def __init__(self, id, pos_x, pos_y):
		self.id = id
		self.x = pos_x
		self.y = pos_y
		self.rect = pygame.draw.rect(screen, GRAY, (self.x, self.y, 100, 100))

	def is_clicked(self, mouse_x, mouse_y):
		# Checks if the field has been clicked
		return True if self.rect.collidepoint(mouse_x, mouse_y) else False

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


# Start application
if __name__ == '__main__':
	pygame.init()
	screen = pygame.display.set_mode(WINDOW)
	pygame.display.set_caption('Tic Tac Toe')
	main()
