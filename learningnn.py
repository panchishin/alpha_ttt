# Basic 3x3 tic tac toe
from random import shuffle, choice
from tabnanny import verbose
from time import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

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

def to_sparse(mini_board_pos):
	"""
	Converts the positions for each player into a 1hot encoding for the NN
	the encoding is 0=empty, 1=player0, 2=player1
	"""
	encoded = np.ones([1,9], dtype=np.int16) * 2
	for index in range(9):
		pos = 1<<index
		if pos == pos & mini_board_pos[0]:
			encoded[0,index] = 0
		elif pos == pos & mini_board_pos[1]:
			encoded[0,index] = 1
	return encoded

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

class MonteCarlo:
	def __init__(self, iterations=50):
		self.iterations = iterations

	def ai(self, mini_board, player):
		avail = ( 0b111111111 ^ ( mini_board.pos[0] | mini_board.pos[1] ) )
		scores = [0,0,0,0,0,0,0,0,0]
		for index in range(9):
			pos = 1 << index
			if pos == avail & pos:
				move = MiniBoard(mini_board)
				move.move(pos, player)
				if move.is_win(player):
					scores[index] += self.iterations * 2
				else:
					for _ in range(self.iterations):
						dup = MiniBoard(move)
						scores[index] -= dup.randgame(1-player)
					scores[index] = scores[index] + self.iterations
		return 1 << scores.index(max(scores))

class NN:

	def __init__(self, other=None):
		if other == None:
			self.model = models.Sequential((
				layers.Input(shape=[9,3], dtype=tf.float32),
				layers.Flatten(),
				layers.Dense(50, activation="tanh"),
				layers.Dense(9)
			))
		else:
			self.model = models.clone_model(other.model)

	def ai(self, mini_board, player):
		# always convert board to be from player 0's point of view
		if player == 0:
			mini_board_pos = mini_board.pos
		else:
			mini_board_pos = mini_board.pos[::-1]

		sparse_board = to_sparse(mini_board_pos=mini_board_pos)
		result = self.model.predict(tf.one_hot(sparse_board,depth=3), verbose=0)

		avail = ( 0b111111111 ^ ( mini_board.pos[0] | mini_board.pos[1] ) )
		for index in range(9):
			pos = 1 << index
			if pos != avail & pos:
				result[0,index] = -1e38 # min float

		return 1 << int(tf.random.categorical(result,1).numpy()[0,0])


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
		print()
	return np.array((0,0))

def fair_battle(ai_a, ai_b, scores):
	scores = scores + battle(ai_a, ai_b, verbose=False)
	scores = scores + battle(ai_b, ai_a, verbose=False)[::-1]
	return scores

nn = NN()
print("\n--- BATTLES ---")
print("monte_carlo_ai vs nn_ai")
scores = np.zeros([2], dtype=np.int16)
for trial in range(1,11):
	for _ in range(5):
		scores = fair_battle(MonteCarlo().ai, nn.ai, scores)
	print("Score is ", scores, "out of", trial*10)

for _ in range(5):
	print("Monto Carlo vs NN")
	print("Result is",battle(MonteCarlo().ai,nn.ai))
	print("NN vs Monto Carlo")
	print("Result is",battle(nn.ai,MonteCarlo().ai))



print("\n--- BATTLES ---")
print("random_ai vs random_ai")
scores = np.zeros([2], dtype=np.int16)
for trial in range(1,11):
	for _ in range(50):
		scores = fair_battle(random_ai, random_ai, scores)
	print("Score is ", scores, "out of", trial*100)

print("\n--- BATTLES ---")
print("rando_ai vs winning_move_ai")
scores = np.zeros([2], dtype=np.int16)
for trial in range(1,11):
	for _ in range(50):
		scores = fair_battle(random_ai, winning_move_ai, scores)
	print("Score is ", scores, "out of", trial*100)

print("\n--- BATTLES ---")
print("nn_ai vs random_ai")
scores = np.zeros([2], dtype=np.int16)
nn = NN()
for trial in range(1,11):
	for _ in range(5):
		scores = fair_battle(nn.ai, random_ai, scores)
	print("Score is ", scores, "out of", trial*10)

print("\n--- BATTLES ---")
print("nn_ai vs winning_move_ai")
scores = np.zeros([2], dtype=np.int16)
nn = NN()
for trial in range(1,11):
	for _ in range(5):
		scores = fair_battle(nn.ai, winning_move_ai, scores)
	print("Score is ", scores, "out of", trial*10)

print("\n--- BATTLES ---")
print("MonteCarlo(50) vs winning_move_ai")
scores = np.zeros([2], dtype=np.int16)
for trial in range(1,11):
	for _ in range(5):
		scores = fair_battle(MonteCarlo(50).ai, winning_move_ai, scores)
	print("Score is ", scores, "out of", trial*10)

print("\n--- BATTLES ---")
print("MonteCarlo(50) vs nn_ai")
scores = np.zeros([2], dtype=np.int16)
for trial in range(1,11):
	for _ in range(5):
		scores = fair_battle(MonteCarlo(50).ai, nn.ai, scores)
	print("Score is ", scores, "out of", trial*10)

print("\n--- BATTLES ---")
print("MonteCarlo(1000) vs nn_ai")
scores = np.zeros([2], dtype=np.int16)
for trial in range(1,11):
	for _ in range(5):
		scores = fair_battle(MonteCarlo(1000).ai, nn.ai, scores)
	print("Score is ", scores, "out of", trial*10)

print("\n--- BATTLES ---")
print("MonteCarlo(50) vs MonteCarlo(1000)")
scores = np.zeros([2], dtype=np.int16)
for trial in range(1,11):
	for _ in range(5):
		scores = fair_battle(MonteCarlo(50).ai, MonteCarlo(1000).ai, scores)
	print("Score is ", scores, "out of", trial*10)
