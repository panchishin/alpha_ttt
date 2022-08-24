
rotate_lookup = [6,3,0,7,4,1,8,5,2]

def rotate_list(items):
    return [items[i] for i in rotate_lookup]

assert([0,1,2,3,4,5,6,7,8]==rotate_list(rotate_list(rotate_list(rotate_list([0,1,2,3,4,5,6,7,8])))))
assert([2,5,8,1,4,7,0,3,6]==rotate_list(rotate_list(rotate_list([0,1,2,3,4,5,6,7,8]))))

def rotate_val(item):
    return 1 << rotate_lookup[int(item).bit_length() - 1]

assert((1<<7) == rotate_val(1<<3))
assert((1<<7) == rotate_val(rotate_val(rotate_val(rotate_val(1<<7)))))
assert((1<<4) == rotate_val(1<<4))

def rotate_training(sparse_boards, reward_list, pos_list, times=1):
	for _ in range(times):
		sparse_boards, reward_list, pos_list = [ rotate_list(x) for x in sparse_boards], [ x for x in reward_list], [ rotate_val(x) for x in pos_list]
	return sparse_boards, reward_list, pos_list