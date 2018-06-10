import pygame
import sys
import numpy as np
import random
import json
from ast import literal_eval
from pygame.locals import *

# Color configuration
GRAY = (245, 245, 245)
BLACK = (41, 40, 48)
RED = (205, 35, 84)
YELLOW = (229, 207, 74)
BLUE = (56, 113, 193)
# PyGame initialization
WINDOW = (400, 550)
pygame.init()
screen = pygame.display.set_mode(WINDOW)
pygame.display.set_caption('Tic Tac Toe')


def play_game():
	while True:
		# Set up screen with restart button
		screen.fill(GRAY)
		font = pygame.font.SysFont("Arial", 32)
		restart = pygame.draw.rect(screen, GRAY, (100, 450, WINDOW[0] / 2, WINDOW[0] / 5))
		screen.blit(font.render("Restart", True, BLACK), (150, 470))
		# Initialize board and players
		player_x = Agent('X')
		player_o = Player('O')
		player_x.load_qtable('sarsa-table.json')
		game_board = Board()
		players = {player_x.id: player_x.mark, player_o.id: player_o.mark}
		current_player = player_x
		play_again = False
		# Timer
		fps = 30
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
					message = font.render("Player {} wins".format(players[game_board.winner]), True, RED)
				screen.blit(message, (100, 400))

			# Event handling during game
			for event in pygame.event.get():
				# Moves of agent
				if isinstance(current_player, Agent):
					move = current_player.select_action(game_board.get_board_state())
					valid_move = game_board.update_board(move, current_player)
					if valid_move:
						current_player = player_o if current_player == player_x else player_x

				# Handle input of human player
				if event.type == QUIT or (event.type == KEYUP and event.key == K_ESCAPE):
					pygame.quit()
					sys.exit()
				elif event.type == MOUSEBUTTONDOWN and event.button == 1:
					mouse_x, mouse_y = event.pos
					# Restart game if restart button is clicked
					play_again = restart.collidepoint(mouse_x, mouse_y)

					if isinstance(current_player, Player):
						move = game_board.get_clicked_field(mouse_x, mouse_y)
						valid_move = game_board.update_board(move, current_player)
						if valid_move:
							current_player = player_o if current_player == player_x else player_x

				# Redisplay
				pygame.display.flip()


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
		pygame.draw.rect(screen, BLACK, (40, 40, 320, 320))
		for row_index, row in enumerate(self.board):
			for col_index, column in enumerate(row):
				x_pos = col_index * 110 + 40
				y_pos = row_index * 110 + 40
				self.fields[(row_index, col_index)] = Field(x_pos, y_pos)

	def get_clicked_field(self, mouse_x, mouse_y):
		# Returns the id of the clicked field
		for field_id, field in self.fields.items():
			if field.is_clicked(mouse_x, mouse_y):
				return field_id
		else:
			return -1, -1

	def update_board(self, field_id, player):
		# Updates the board when a field has been selected
		valid_move = False
		if field_id == (-1, -1):
			pass
		elif self.board[field_id] == 0:
			valid_move = True
			self.board[field_id] = player.id
			field = self.fields[field_id]
			field.draw_mark(player.id)
		return valid_move

	def get_board_state(self):
		# An array cannot be a dictionary key, the board state is therefore returned as tuple
		return tuple(map(tuple, self.board))

	def gameover(self):
		# Checks if a row, column or diagonal has a winning constellation
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


class Agent(Player):
	# Selects a random action
	def __init__(self, mark):
		Player.__init__(self, mark)
		self.qtable = {}

	@staticmethod
	def possible_actions(state):
		# Return all empty fields in the given fields which are available as actions
		actions = []
		for row in range(3):
			for col in range(3):
				if state[row][col] == 0:
					actions.append((row, col))
		return actions

	def get_qvalue(self, state, action):
		# Return Q-value for state-action pair if it exists, otherwise start with a Q-value of 0
		# in order to encourage exploration of new states
		if (state, action) not in self.qtable:
			self.qtable[(state, action)] = 0
		return self.qtable[(state, action)]

	def select_action(self, board_state):
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
		return move

	def load_qtable(self, file):
		with open(file) as json_data:
			table = json.load(json_data)
		for key, value in table.items():
			self.qtable[literal_eval(key)] = value


# Start application
if __name__ == '__main__':
	play_game()
