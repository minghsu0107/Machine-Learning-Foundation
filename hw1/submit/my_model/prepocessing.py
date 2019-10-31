import numpy as np
import random

def load(filename):
	x = []
	y = []
	with open(filename, 'r', encoding='UTF-8') as file:
		for line in file:
			tmp_list = list(map(float, line.split()))
			x.append(np.array([float(1)] + [i for i in tmp_list[:4]]))
			y.append(int(tmp_list[4]))
	return x, y

def shuffle_data(x, y, seed):
	tmp = list(zip(x, y))
	random.seed(seed)
	random.shuffle(tmp)
	return zip(*tmp)

def seed_generator(t):
	return t ** 17