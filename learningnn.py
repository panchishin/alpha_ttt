# Basic 3x3 tic tac toe
from random import shuffle
from time import time

winning_configs = (
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
		for config in winning_configs:
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
	
	def montecarlo(self, player, iterations=1000):
		start = time()
		avail = ( 0b111111111 ^ ( self.pos[0] | self.pos[1] ) )
		scores = [0,0,0,0,0,0,0,0,0]
		for index in range(9):
			pos = 1 << index
			if pos == avail & pos:
				move = MiniBoard(self)
				move.move(pos, player)
				for _ in range(iterations):
					dup = MiniBoard(move)
					scores[index] -= dup.randgame(1-player)
				scores[index] = scores[index] + iterations
		print(f'elapse {time() - start}')
		return scores

	def print(self):
		print(f'{self.pos[0]:>09b} player 0')
		print(f'{self.pos[1]:>09b} player 1')



board = MiniBoard()
board.print()
for index,score in zip(range(9), board.montecarlo(0)):
	print(f'{(1<<index):>09b} {score:5}')

board.pos[0] = 0b100000100
board.pos[1] = 0b001100000

board.print()
for index,score in zip(range(9), board.montecarlo(0)):
	if score != 0.:
		print(f'{(1<<index):>09b} {score:5}')
