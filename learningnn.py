# Basic 3x3 tic tac toe
from random import shuffle, choice
from time import time
import numpy as np

winning_player_board = (
	0b111000000, 0b000111000, 0b000000111,
	0b100100100, 0b010010010, 0b001001001,
	0b100010001, 0b001010100
)

legal_positions = list([1<<i for i in range(9)])

class MiniBoard:
	__slots__ = ['pos']

	def __init__(self, other=None):
		if other is None:
			self.pos = [0,0]
		else:
			self.pos = [other.pos[0], other.pos[1]]
	
	def move(self, pos, player):
		self.pos[player] |= pos
	
	def is_win(self, player):
		for config in winning_player_board:
			if config == self.pos[player] & config:
				return 1
		return 0

	def _randpos(self):
		avail = ( 0b111111111 ^ ( self.pos[0] | self.pos[1] ) )
		shuffle(legal_positions)
		for pos in legal_positions:
			if pos == avail & pos:
				return pos
		return 0
	
	def randgame(self, player):
		if self.is_win(player):
			return 1
		current = player
		pos = self._randpos()
		while pos != 0:
			self.move(pos, current)
			if self.is_win(current):
				return 1 if current == player else -1
			current = 1 - current
			pos = self._randpos()
		return 0

	def print(self):
		for row in range(3):
			if row > 0:
				print("---+---+---")
			print(" ", end="")
			for col in range(3):
				if col > 0:
					print(" | ", end="")
				pos = ( 1 << col ) << (3*row)
				if pos == self.pos[0] & pos :
					print("X", end="")
				elif pos == self.pos[1] & pos :
					print("O", end="")
				else:
					print(" ", end="")
				
			print()

	def to_one_hot(self):
		"""
		Converts the positions for each player into a 1hot encoding for the NN
		the encoding is 0=empty, 1=player0, 2=player1
		"""
		encoded = np.ones([9], dtype=np.int16) * 2
		for index in range(9):
			pos = 1<<index
			if pos == pos & self.pos[0]:
				encoded[index] = 0
			elif pos == pos & self.pos[1]:
				encoded[index] = 1
		return encoded

	def from_one_hot(self, one_hot):
		self.pos = [0,0]
		for index in range(9):
			pos = 1<<index
			if one_hot[index] == 0:
				self.pos[0] = self.pos[0] | pos
			if one_hot[index] == 1:
				self.pos[1] = self.pos[1] | pos


def random_ai(mini_board, player):
	avail = ( 0b111111111 ^ ( mini_board.pos[0] | mini_board.pos[1] ) )
	candidate = []
	for pos in legal_positions:
		if pos == avail & pos:
			candidate.append(pos)
	return choice(candidate)

def winning_move_ai(mini_board, player):
	avail = ( 0b111111111 ^ ( mini_board.pos[0] | mini_board.pos[1] ) )
	candidate = []
	for pos in legal_positions:
		if pos == avail & pos:
			move = MiniBoard(mini_board)
			move.move(pos, player=player)
			if move.is_win(player):
				return pos
			candidate.append(pos)
	return choice(candidate)

def montecarlo_ai(mini_board, player, iterations=50):
	avail = ( 0b111111111 ^ ( mini_board.pos[0] | mini_board.pos[1] ) )
	scores = [0,0,0,0,0,0,0,0,0]
	for index in range(9):
		pos = 1 << index
		if pos == avail & pos:
			move = MiniBoard(mini_board)
			move.move(pos, player)
			if move.is_win(player):
				scores[index] += iterations * 2
			else:
				for _ in range(iterations):
					dup = MiniBoard(move)
					scores[index] -= dup.randgame(1-player)
				scores[index] = scores[index] + iterations
	return 1 << scores.index(max(scores))

def battle(ai_a, ai_b, verbose=True):
	"""
	returns a numpy array of length 2 reporting the number of wins
	"""
	mini_board = MiniBoard()

	ai_player = (ai_a, ai_b)

	for turn in range(9):
		player = turn % 2
		pos = ai_player[player](mini_board,player)
		mini_board.move(pos=pos, player=player)
		if verbose:
			mini_board.print()
		if mini_board.is_win(player=player) :
			return np.array((1,0)) if player == 0 else np.array((0,1))

	if verbose:
		mini_board.print()
	return np.array((0,0))

def fair_battle(ai_a, ai_b, scores):
	scores = scores + battle(ai_a, ai_b, verbose=False)
	scores = scores + battle(ai_b, ai_a, verbose=False)[::-1]
	return scores

battle(random_ai, montecarlo_ai)


print("\n--- BATTLES ---")
print("Random vs winning_move_ai")
for _ in range(10):
	scores = np.zeros([2], dtype=np.int16)
	for _ in range(5):
		scores = fair_battle(random_ai, winning_move_ai, scores)
	print("Score is ", scores, "out of 10")

print("\n--- BATTLES ---")
print("Montecarlo vs winning_move_ai")
for _ in range(10):
	scores = np.zeros([2], dtype=np.int16)
	for _ in range(5):
		scores = fair_battle(montecarlo_ai, winning_move_ai, scores)
	print("Score is ", scores, "out of 10")

