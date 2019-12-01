import numpy as np
import matplotlib.pyplot as plt
import argparse

def generate(n, prob=0.2):
	rng = np.random.RandomState(seed=None)
	X = rng.uniform(-1, 1, size=n)
	y = np.sign(X)
	y[np.random.uniform(0, 1, size=n) < prob] *= -1
	return X, y

def get_theta(sorted_X):
	theta = (sorted_X[1:] + sorted_X[:-1]) / 2
	theta = np.r_[theta, sorted_X[0] - 1]
	theta = np.r_[theta, sorted_X[-1] + 1]
	return theta.reshape(-1, 1)

def plotHist(my_list, xlabel=None, title=None, bins=None):
	plt.figure()
	ax = plt.axes()
	if bins != None:
		ax.hist(my_list, facecolor='orange', alpha=0.8, bins=bins)
	else:
		ax.hist(my_list, facecolor='orange', alpha=0.8)
	ax.set_xlabel(xlabel)
	ax.set_title(title)
	plt.show()

def stump(X, y):
	sortedX = np.sort(X)
	theta = get_theta(sortedX)
	X = np.tile(X, (theta.shape[0], 1)) # repeat X for theta.shape[0] times

	min_err = 1
	for s in [-1, 1]:
		error = np.sum(s * np.sign(X - theta) != y, axis=1) #(y.shape[0],)
		min_idx = np.argmin(error, axis=0)
		if (error[min_idx] / y.shape[0] < min_err):
			min_err = error[min_idx] / y.shape[0]
			_s = s
			_t = theta[min_idx][0]
	return _s, _t, min_err 

def test(n):
	EIN = []
	EOUT = []
	EIN_MINUS_EOUT = []
	for i in range(1000):
		print("training round {}".format(i + 1))
		X, y = generate(n)
		s, theta, min_ein = stump(X, y)
		eout = 0.5 + 0.3 * s * (np.abs(theta) - 1)
		EIN.append(min_ein)
		EOUT.append(eout)
		EIN_MINUS_EOUT.append(min_ein - eout)

	plotHist(EIN_MINUS_EOUT, xlabel="Ein-Eout", title="Ein-Eout Histogram", bins=20)

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-n", "--size", required=True,
					help="size of data")
	args = vars(ap.parse_args())

	n = int(args["size"])
	test(n)