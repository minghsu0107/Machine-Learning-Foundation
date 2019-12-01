import numpy as np
import matplotlib.pyplot as plt

def my_load(filename):
	data = np.genfromtxt(filename)
	X = data[:, :-1] #(n, m)
	y = data[:, -1] #(n,)
	return X, y

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

def multidimension_stump(X, y):
	m = X.shape[1]
	_s = 0
	_theta = 0
	_dim = 0
	min_ein = 1
	for i in range(m):
		s, theta, ein = stump(X[:, i], y)
		if ein < min_ein:
			min_ein = ein
			_s = s
			_theta = theta
			_dim = i
	return _s, _theta, _dim, min_ein

def test():
	EIN = []
	EOUT = []
	EIN_MINUS_EOUT = []
	for i in range(1000):
		print("training round {}".format(i + 1))
		X, y = generate(20)
		s, theta, min_ein = stump(X, y)
		'''
		to get min(eout):
		theta = 0
		s = 1
		(By mathematically computing the minium of the eout equation)
		-> min(eout) = 0.2
		'''
		eout = 0.5 + 0.3 * s * (np.abs(theta) - 1)
		EIN.append(min_ein)
		EOUT.append(eout)
		EIN_MINUS_EOUT.append(min_ein - eout)

	plotHist(EIN, xlabel="Ein", title="Ein Histogram", bins=20)
	plotHist(EOUT, xlabel="Eout", title="Eout Histogram", bins=20)
	plotHist(EIN_MINUS_EOUT, xlabel="Ein-Eout", title="Ein-Eout Histogram", bins=20)

def test2():
	X, y = my_load("../../hw1/submit/my_data/pocket_train.dat")
	print(X.shape, y.shape)
	s, theta, dim, min_ein = multidimension_stump(X, y)
	print(s, theta, dim, min_ein, min_ein)
	print(np.sum(s * np.sign(X[:, dim] - theta) != y) / y.shape[0]) # == min_ein

if __name__ == "__main__":
	test()