# Basic 3x3 tic tac toe
from random import shuffle, choice
from tabnanny import verbose
from time import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# CHANGE LOG
# use exponential decay for reward function

train_games = 2000
eval_games = 200
epochs = 100


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
				layers.Dense(200, activation="relu"),
				layers.Dense(50, activation="relu"),
				layers.Dense(9, activation="softmax")
			))
		else:
			self.model = models.clone_model(other.model)
		self.model.compile(loss="mae", optimizer="adam")

	def save(self, name):
		self.model.save(name)
	
	def load(self, name):
		self.model = models.load_model(name)

	def self_play(self, games=1):
		sparse_boards = []
		logits = []
		pos_list = []

		if games > 1:
			for game in range(1,games+1):
				s, l = self.self_play()
				sparse_boards.extend(s)
				logits.extend(l)
				if game%5 == 0:
					print(".",end="", flush=True)
				if game%500 == 0:
					print(" -",game,"games", flush=True)
			return np.array(sparse_boards), np.array(logits)

		mini_board = MiniBoard()

		for turn in range(9):
			player = turn % 2
			sparse_board, logit, pos = self.ai(mini_board=mini_board, player=player, report_logits=True)

			sparse_boards.append(sparse_board[0].tolist())
			logits.append(logit.tolist())
			pos_list.append(pos)

			mini_board.move(pos=pos, player=player)
			if verbose:
				mini_board.print()
			if mini_board.is_win(player=player) :
				for move in range(turn):
					learning_rate = 0.5**(turn-move) if move%2==player else -(0.5**(turn-move))
					index = pos_list[move].bit_length() - 1
					logits[move][index] += learning_rate
					if np.min(logits[move]) < 0:
						logits[move] = logits[move] - np.min(logits[move])
					logits[move] = logits[move] / np.sum(logits[move])
				break

		return sparse_boards, logits

	def train(self, sparse_boards, target_logits):
		x = tf.one_hot(sparse_boards,depth=3)
		y = target_logits
		dataset = tf.data.Dataset.from_tensor_slices((x, y)).repeat(epochs).shuffle(1000).batch(32)
		self.model.evaluate(x, y)
		self.model.fit(dataset)
		self.model.evaluate(x, y)

	def ai(self, mini_board, player, report_logits=False):
		# always convert board to be from player 0's point of view
		if player == 0:
			mini_board_pos = mini_board.pos
		else:
			mini_board_pos = mini_board.pos[::-1]

		sparse_board = to_sparse(mini_board_pos=mini_board_pos)
		predict = self.model.predict(tf.one_hot(sparse_board,depth=3), verbose=0)[0]
		if report_logits:
			logits = np.array(predict)

		avail = ( 0b111111111 ^ ( mini_board.pos[0] | mini_board.pos[1] ) )
		for index in range(9):
			pos = 1 << index
			if pos != avail & pos:
				predict[index] = 0

		predict = predict * predict
		predict = predict / sum(predict)
		pos = 1 << int(np.random.choice(range(9),p=predict))

		if report_logits:
			return sparse_board, logits, pos
		else:
			return pos




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

def report_fair_battle(ai_names, ai_funcs, trials, iterations):
	iterations = iterations // 2
	scores = np.zeros([2], dtype=np.int16)
	for trial in range(1,trials+1):
		for _ in range(iterations):
			scores = fair_battle(ai_funcs[0], ai_funcs[1], scores)
		print("Fair battle :", ai_names[0], "vs", ai_names[1], "Score is", scores, "out of", trial*iterations*2)


nn = NN()
nn.save(name=f"saves/nn_0")
# nn.load(name=f"saves/nn_3")
nn_old = NN(nn)
# nn_old.load(name=f"saves/nn_0")
print("\nUntrained NN baseline")
report_fair_battle(["nn.ai","random_ai"], [nn.ai, random_ai], 1, eval_games)
report_fair_battle(["nn.ai","nn_old.ai"], [nn.ai, nn_old.ai], 1, eval_games)

for training_session in range(1,21):
	print(f"Self play {train_games} games, training session", training_session)
	sparse_boards, logits = nn.self_play(train_games)
	nn.train(sparse_boards=sparse_boards, target_logits=logits)
	# nn.save(name=f"saves/nn_{training_session}")
	report_fair_battle(["nn.ai","random_ai"], [nn.ai, random_ai], 1, eval_games)
	report_fair_battle(["nn.ai","nn_old.ai"], [nn.ai, nn_old.ai], 1, eval_games)
