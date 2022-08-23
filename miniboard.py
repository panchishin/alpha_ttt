from random import shuffle

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

	def available(self):
		return ( 0b111111111 ^ ( self.pos[0] | self.pos[1] ) )

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

