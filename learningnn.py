import numpy as np
from miniboard import MiniBoard
from basic_ai import MonteCarlo, random_ai, winning_move_ai
from nn_ai import NN
from rotate import rotate_training
from random import randint

train_games = 50
eval_games = 20
epochs = 1

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
		print(f"Fair battle : {scores[0]:3} {ai_names[0]:10} vs {scores[1]:3} {ai_names[1]:19} and {(trial*iterations*2-scores[0]-scores[1]):3} ties out of", trial*iterations*2, "games")



nn = NN()
# nn.save(name=f"saves/nn_0")
# nn.load(name=f"saves/nn_3")
print("\nUntrained NN baseline")
nn.print_first_move_expectation()
report_fair_battle(["nn.ai","random_ai"],[(lambda b,p: nn.ai(b,p)[2]),random_ai],1,eval_games)
report_fair_battle(["nn.ai","winning_move_ai"],[(lambda b,p: nn.ai(b,p)[2]),winning_move_ai],1,eval_games)
report_fair_battle(["nn.ai","MonteCarlo(250).ai"],[(lambda b,p: nn.ai(b,p)[2]),MonteCarlo(250).ai],1,eval_games)

for round in range(1,21):
	print(end=f"Round {round} Training {train_games} simulations :", flush=True)
	for i in range(train_games):
		if i%50==0 : print(end=f" {i} ", flush=True)
		elif i%5==0 : print(end=".", flush=True)
		sparse_boards, reward_list, index_list = nn.self_play(1)
		reward_list = [x[i] for x,i in zip(reward_list, index_list)]
		# sparse_boards, reward_list, index_list = rotate_training(sparse_boards, reward_list, index_list, randint(0,3))
		nn.train(sparse_boards, reward_list, index_list, epochs=epochs)
	print(" Training complete.")
	nn.print_first_move_expectation()
	report_fair_battle(["nn.ai","random_ai"],[(lambda b,p: nn.ai(b,p)[2]),random_ai],1,eval_games)
	report_fair_battle(["nn.ai","winning_move_ai"],[(lambda b,p: nn.ai(b,p)[2]),winning_move_ai],1,eval_games)
	report_fair_battle(["nn.ai","MonteCarlo(250).ai"],[(lambda b,p: nn.ai(b,p)[2]),MonteCarlo(250).ai],1,eval_games)
