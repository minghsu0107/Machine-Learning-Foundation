import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import argparse
from my_model.prepocessing import load, shuffle_data, seed_generator
from my_model.model import PLA

def p6():
	X, Y = load('my_data/PLA.dat')
	updates = []
	for i in range(1126):
		p = PLA(shuffle=shuffle_data, pocket=False, 
				random_state=seed_generator(i))
		p.fit(X, Y)
		updates.append(p.n_updates)
		print("training round {}".format(i + 1))
	
	updates = sorted(updates)
	plt.figure()
	ax = plt.axes()
	ax.hist(updates, # bins=(updates[-1] - updates[0] + 1), 
			facecolor='orange', alpha=0.8)
	ax.set_xlabel('Number of updates')
	ax.set_ylabel('Frequency')
	ax.set_title('Average number of updates is {}'.format(int(sum(updates) / len(updates))))
	plt.show()

def p7():
	X, Y = load('my_data/pocket_train.dat')
	test_X, test_Y = load('my_data/pocket_test.dat')

	errors = []
	for i in range(1126):
		p = PLA(max_iter=100, shuffle=shuffle_data, pocket=True,
				random_state=seed_generator(i))
		p.fit(X, Y)
		errors.append(1 - p.score(test_X, test_Y))
		print("training round {}".format(i + 1))
	
	plt.figure()
	ax = plt.axes()
	ax.hist(errors, facecolor='blue', alpha=0.8)
	ax.set_xlabel('Error rate')
	ax.set_ylabel('Frequency')
	ax.set_title('Average error rate is {}'.format(np.array(errors).mean()))
	plt.show()

def p8():
	X, Y = load('my_data/pocket_train.dat')
	test_X, test_Y = load('my_data/pocket_test.dat')

	errors = []
	for i in range(1126):
		p = PLA(max_iter=100, shuffle=shuffle_data, pocket=False,
				random_state=seed_generator(i))
		p.fit(X, Y)
		errors.append(1 - p.score(test_X, test_Y))
		print("training round {}".format(i + 1))
	
	plt.figure()
	ax = plt.axes()
	ax.hist(errors, facecolor='green', alpha=0.8)
	ax.set_xlabel('Error rate')
	ax.set_ylabel('Frequency')
	ax.set_title('Average error rate is {}'.format(np.array(errors).mean()))
	plt.show()

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--problem", required=True,
					help="Problem to be solved")
	args = vars(ap.parse_args())

	prob = args["problem"]
	if prob == '6':
		p6()
	elif prob == '7':
		p7()
	elif prob == '8':
		p8()
	else:
		print("Invalid question number")