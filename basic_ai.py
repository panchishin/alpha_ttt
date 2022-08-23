from miniboard import legal_positions, MiniBoard
from random import choice

def random_ai(mini_board, player):
	avail = mini_board.available()
	candidate = []
	for pos in legal_positions:
		if pos == avail & pos:
			candidate.append(pos)
	return choice(candidate)

def winning_move_ai(mini_board, player):
	avail = mini_board.available()
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
		avail = mini_board.available()
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
