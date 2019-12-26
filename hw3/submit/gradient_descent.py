import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.simplefilter('ignore')

def clock(func):
    def clocked(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print('[%0.8fs]' % (elapsed))
        return result
    return clocked

def add_intercept(X):
	n = X.shape[0]
	return np.c_[np.ones(n), X]

def my_load(filename):
	data = np.genfromtxt(filename)
	X = data[:, :-1] # shape = (n, m)
	X = add_intercept(X) # shape = (n, m + 1)
	y = data[:, -1].astype(int) # shape = (n,)
	y = y.reshape(-1, 1) # shape = (n, 1)
	return X, y

def sigmoid(s):
	return 1 / (np.exp(-s) + 1)

def gradient(X, w, y):
	tmp1 = sigmoid(-X.dot(w) * y) # shape = (X.shape[0], X.shape[1])
	tmp2 = -X * y # shape = (X.shape[0], X.shape[1])
	grad = np.mean(tmp1 * tmp2, axis=0).reshape(-1, 1)
	return grad

def next_batch(X, y, batch_size):
	for i in range(0, X.shape[0], batch_size):
		yield (X[i:i + batch_size], y[i:i + batch_size].reshape(-1, 1))

@clock
def GD(X, y, max_iter=2000, eta0=0.01, random_initialize=True, 
	   random_state=None, **kwargs):
	if random_initialize == True:
		rng = np.random.RandomState(seed=random_state)
		w = rng.uniform(low=0.0, high=1.0, size=(X.shape[1], 1))
	else:
		w = np.zeros((X.shape[1], 1))

	for i in range(max_iter):
		grad = gradient(X, w, y)
		# w -= np.multiply(eta0, grad)
		w -= eta0 * grad

	return w

# In general, the mini-batch size is not a hyperparameter that 
# you should worry much about. You basically determine 
# how many training examples will fit on your GPU/main memory 
# and then use the nearest power of 2 as the batch size
@clock
def SGD(X, y, max_iter=2000, eta0=0.01, random_initialize=True, random_state=None, 
		batch_size=1):
	
	if random_initialize == True:
		rng = np.random.RandomState(seed=random_state)
		w = rng.uniform(low=0.0, high=1.0, size=(X.shape[1], 1))
	else:
		w = np.zeros((X.shape[1], 1))

	training_data = np.hstack((X, y))
	rng.shuffle(training_data)
	X = training_data[:, :-1]
	y = training_data[:, -1]

	i = 0
	while i < max_iter:
		for (batchX, batchY) in next_batch(X, y, batch_size):
			grad = gradient(batchX, w, batchY)
			w -= np.multiply(eta0, grad)
			i += 1
			if i == max_iter:
				break

	return w


def solve1(X_train, y_train, X_test, y_test, 
		  max_iter=2000, eta0=0.01, random_initialize=True, random_state=None,
		  sgd=True, batch_size=1, **kwargs):

	if sgd == True:
		w = SGD(X_train, y_train, max_iter, eta0, random_initialize, 
			    random_state, batch_size)
	else:
		w = GD(X_train, y_train, max_iter, eta0, random_initialize, random_state)

	y_train_pred = X_train.dot(w)
	y_train_pred[y_train_pred > 0] = 1
	y_train_pred[y_train_pred <= 0] = -1

	y_test_pred = X_test.dot(w)
	y_test_pred[y_test_pred > 0] = 1
	y_test_pred[y_test_pred <= 0] = -1

	Ein = np.mean(y_train_pred != y_train)
	Eout = np.mean(y_test_pred != y_test)
	if sgd == True:
		print("SGD: Ein = {}, Eout = {}\n".format(Ein, Eout))
	else:
		print("GD: Ein = {}, Eout = {}\n".format(Ein, Eout))

	return Ein, Eout

def solve2(X_train, y_train, X_test, y_test, random_state=None):
	clfs = [Perceptron(max_iter=1000, random_state=random_state),
			SGDClassifier(max_iter=1000, loss='log', random_state=random_state),
			RandomForestClassifier(random_state=random_state)]
	clf_names = ['Perceptron', 'SGD Classifier', 'Random Forest']

	y_train = y_train.ravel()
	y_test = y_test.ravel()

	@clock
	def run(clf):
		clf.fit(X_train, y_train)

	for i, clf in enumerate(clfs):
		run(clf)
		Ein = 1 - clf.score(X_train, y_train)
		Eout = 1 - clf.score(X_test, y_test)
		print("{}: Ein = {}, Eout = {}\n".format(clf_names[i], Ein, Eout))

def test(X_train, y_train, X_test, y_test):
	T = int(sys.argv[1]) if len(sys.argv) == 2 else 2000
	D = len(X_train[0])
	w_GD = np.zeros((D, 1))
	w_SGD = np.zeros((D, 1))

	Ein_GD = np.zeros(T)
	Ein_SGD = np.zeros(T)
	Eout_GD = np.zeros(T)
	Eout_SGD = np.zeros(T)

	eta_GD = 0.01
	eta_SGD = 0.01

	for t in range(T):
		# fixed learning rate gradient descent
		w_GD = w_GD - eta_GD * gradient(X_train, w_GD, y_train)
		y_train_pred = X_train.dot(w_GD)
		y_train_pred[y_train_pred > 0] = 1
		y_train_pred[y_train_pred <= 0] = -1

		y_test_pred = X_test.dot(w_GD)
		y_test_pred[y_test_pred > 0] = 1
		y_test_pred[y_test_pred <= 0] = -1

		Ein_GD[t]  = np.mean(y_train_pred != y_train)
		Eout_GD[t] = np.mean(y_test_pred != y_test)

		# stochastic gradient descent
		x, y = X_train[t % X_train.shape[0], :].reshape(1, -1), y_train[t % X_train.shape[0], :].reshape(1, -1)
		w_SGD = w_SGD - eta_SGD * gradient(x, w_SGD, y)
		y_train_pred = X_train.dot(w_SGD)
		y_train_pred[y_train_pred > 0] = 1
		y_train_pred[y_train_pred <= 0] = -1

		y_test_pred = X_test.dot(w_SGD)
		y_test_pred[y_test_pred > 0] = 1
		y_test_pred[y_test_pred <= 0] = -1

		Ein_SGD[t]  = np.mean(y_train_pred != y_train)
		Eout_SGD[t] = np.mean(y_test_pred != y_test)

	t_axis = [t for t in range(T)]
	plt.figure()
	plt.plot(t_axis, Ein_GD, 'g', label = "GD")
	plt.plot(t_axis, Ein_SGD, 'b', label = "SGD")
	plt.title("$E_{in}(\mathbf{w}_t) : t$")
	plt.xlabel("$t$")
	plt.ylabel("$E_{in}(\mathbf{w}_t)$")
	plt.legend()
	plt.show()

	plt.figure()
	plt.plot(t_axis, Eout_GD, 'g', label = "GD")
	plt.plot(t_axis, Eout_SGD, 'b', label = "SGD")
	plt.title("$E_{out}(\mathbf{w}_t) : t$")
	plt.xlabel("$t$")
	plt.ylabel("$E_{out}(\mathbf{w}_t)$")
	plt.legend()
	plt.show()

if __name__ == "__main__":
	X_train, y_train = my_load("my_data/train.dat")
	X_test, y_test = my_load("my_data/test.dat")

	print(X_train.shape, y_train.shape)
	unique_train, counts_train = np.unique(y_train, return_counts=True)
	unique_test, counts_test = np.unique(y_test, return_counts=True)

	print("Data Overview\n")
	for item in dict(zip(unique_train, counts_train)).items():
		print("Train label {} counts: {}".format(item[0], item[1]))
	for item in dict(zip(unique_test, counts_test)).items():
		print("Test label {} counts: {}".format(item[0], item[1]))
	print("\nResults:\n")

	params = {'X_train': X_train,
			  'y_train': y_train,
			  'X_test': X_test,
			  'y_test': y_test,
			  'sgd': False,
			  'eta0': 0.01,
			  'max_iter': 2000,
			  'random_state': 50}
	
	solve1(**params)
	solve1(X_train, y_train, X_test, y_test, 
		  sgd=True, batch_size=1, max_iter=10000, eta0=0.01, random_state=50)
	solve2(X_train, y_train, X_test, y_test, random_state=50)
	
	test(X_train, y_train, X_test, y_test)