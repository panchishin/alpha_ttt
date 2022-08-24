import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from miniboard import MiniBoard


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

class OneHot(tf.keras.layers.Layer):
	def __init__(self, depth=3):
		super(OneHot, self).__init__()
		self.depth = depth

	def call(self, inputs):
		return tf.one_hot(inputs, depth=self.depth)

class FilterPolicyByPosition(tf.keras.layers.Layer):

	def __init__(self, depth=9):
		super(FilterPolicyByPosition, self).__init__()
		self.depth = depth

	def call(self, inputs):
		policy, position_onehot = inputs
		policy_with_only_position = tf.multiply(policy,position_onehot)
		return tf.math.reduce_sum(policy_with_only_position,axis=-1)

class NN:

	def __init__(self):
		self.q = 0.5
		
		board_input = layers.Input(shape=[9], dtype=tf.int32, name="board_input")
		position_input = layers.Input(shape=[], dtype=tf.int32, name="position_input")

		onehot = layers.Lambda(lambda x: tf.one_hot(x,3))(board_input)
		flatten = layers.Flatten(name="input")(onehot)
		hidden = layers.Dense(32, activation="tanh")(flatten)
		hidden = layers.BatchNormalization()(hidden)
		policy = layers.Dense(9, activation="tanh", name="policy")(hidden)

		position_onehot = layers.Lambda(lambda x: tf.one_hot(x,9))(position_input)
		policy_with_only_position = layers.Lambda(lambda x: tf.multiply(x[0],x[1]))((policy,position_onehot))
		policy_single = layers.Lambda(lambda x: tf.math.reduce_sum(x,axis=-1))(policy_with_only_position)

		self.model = tf.keras.Model(inputs=board_input, outputs=policy)
		self.model.compile(loss="mae", optimizer="sgd")
		self.train_model = tf.keras.Model(inputs=[board_input,position_input], outputs=[policy_single])
		self.train_model.compile(loss="mae", optimizer="adam")

	def save(self, name):
		self.model.save(name)
	
	def load(self, name):
		self.model = models.load_model(name)

	def self_play(self, games=1):

		if games > 1:
			sparse_boards = []
			reward_list = []
			pos_list = []
			for game in range(games):
				s, r, p = self.self_play()
				r = [x[i] for x,i in zip(r, p)]
				sparse_boards.extend(s)
				reward_list.extend(r)
				pos_list.extend(p)
				if game%50 == 0:
					print(f" {game} ",end="", flush=True)
				elif game%5 == 0:
					print(".",end="", flush=True)

			return sparse_boards, reward_list, pos_list

		else:
			return self.self_play_one_game()

	def self_play_one_game(self, verbose=False):
		mini_board = MiniBoard()
		sparse_boards = []
		reward_list = []
		index_list = []

		for turn in range(9):
			player = turn % 2
			sparse_board, reward, index = self.ai(mini_board=mini_board, player=player, exploration=0.5)
			sparse_boards.append(sparse_board[0].tolist())
			reward_list.append(reward)
			index_list.append(index)

			mini_board.move(pos=(1<<index), player=player)
			if verbose:
				mini_board.print()
			if mini_board.is_win(player=player) :
				for move in range(turn):
					learning_rate = self.q**(turn-move)
					value = 1 if (move%2)==player else -1
					reward_list[move][index_list[move]] = learning_rate * value + (1-learning_rate) * reward_list[move][index_list[move]]
				return sparse_boards, reward_list, index_list

		for move in range(9):
			learning_rate = self.q**(8-move)
			value = 0
			reward_list[move][index_list[move]] = (1-learning_rate) * reward_list[move][index_list[move]]

		return sparse_boards[:-1], reward_list[:-1], index_list[:-1]

	def train(self, sparse_boards, reward_list, pos_list, epochs):
		s = np.array(sparse_boards, dtype=np.int32)
		p = np.array(pos_list, dtype=np.int32)
		r = np.array(reward_list, dtype=np.float32)
		dataset = tf.data.Dataset.from_tensor_slices(((s,p), r)).repeat(epochs).shuffle(10000).batch(32)
		self.train_model.fit(dataset, verbose=0)

	def ai(self, mini_board, player, exploration=0.0):
		# always convert board to be from player 0's point of view
		if player == 0:
			mini_board_pos = mini_board.pos
		else:
			mini_board_pos = mini_board.pos[::-1]

		sparse_board = to_sparse(mini_board_pos=mini_board_pos)
		predict = self.model.predict(sparse_board, verbose=0)[0]

		avail = mini_board.available()
		winning_move = -1
		for index in range(9):
			pos = 1 << index
			if pos != avail & pos:
				predict[index] = -1.0001 # just a little lower than tanh can output
			else:
				move = MiniBoard(mini_board)
				move.move(pos, player=0)
				if move.is_win(0):
					winning_move = pos.bit_length() - 1

		if winning_move != -1:
			index = winning_move
		else:
			choice = np.array(predict) + np.random.normal(scale=exploration, size=[9])
			index = np.argmax(choice)
		return sparse_board, predict, index

	def print_first_move_expectation(self):
		_, predict, _ = self.ai(MiniBoard(), player=0)
		for i in range(0,9,3):
			print(" | ".join([f"{int(x*1000):4}" for x in predict[i:i+3]]))



def show_one_game(nn):
    player = 0
    mini_board = MiniBoard()
    for _ in range(4):
        _, predict, index = nn.ai(mini_board, player=player)
        print("player", player, "moves to", index)
        for i in range(0,9,3):
            print(" ".join([f"{int(x*1000):4}" for x in predict[i:i+3]]))
        mini_board = MiniBoard(mini_board)
        mini_board.move(pos=(1<<index), player=player)
        mini_board.print()
        player = 1-player


def show_one_selfplay(nn):
    sparse_boards, reward_list, pos_list = nn.self_play(1)
    player = 0
    mini_board = MiniBoard()
    for _, predict, index in zip(sparse_boards, reward_list, pos_list):
        print("player", player, "moves to", index)
        for i in range(0,9,3):
            print(" ".join([f"{int(x*1000):4}" for x in predict[i:i+3]]))
        mini_board = MiniBoard(mini_board)
        mini_board.move(pos=(1<<index), player=player)
        mini_board.print()
        player = 1-player


if __name__ == "__main__":
	nn = NN()
	show_one_game(nn)

	tf.keras.utils.plot_model(nn.train_model, show_shapes=True)

	for _ in range(20):
		sparse_boards, reward_list, pos_list = nn.self_play(1)
		reward_list = [x[i] for x,i in zip(reward_list, pos_list)]
		nn.train(sparse_boards, reward_list, pos_list)

	sparse_boards, reward_list, pos_list = nn.self_play_one_game()
	nn.train(sparse_boards, reward_list, pos_list)
	sparse_boards, reward_list, pos_list = nn.self_play(50)
	nn.train(sparse_boards, reward_list, pos_list)

