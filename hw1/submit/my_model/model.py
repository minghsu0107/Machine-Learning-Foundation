import numpy as np
import random
class PLA():
	def __init__(self, max_iter=1000, eta0=1.0, pocket=False, shuffle=random.shuffle, 
				random_state=None):
		self.max_iter = max_iter
		self.eta0 = eta0
		self.pocket = pocket
		self.shuffle = shuffle
		self.random_state = random_state
		self.coef_ = []
		self.n_updates = 0

		if not hasattr(self.shuffle, '__call__'):
			raise ValueError('shuffle should be callable')
		if (not (type(self.random_state) is int)) and (not self.random_state is None):
			raise ValueError('random_state should be an integer')
	
	def check(self, X, Y=None):
		if len(np.array(X).shape) != 2:
			raise ValueError('Wrong input shape of X: {}'.format(X.shape)) 
		if Y != None:
			if len(X) != len(Y):
				raise ValueError('Shape of X does not match Y')

	def count_error(self, X, Y, w):
		return sum(np.sign(X.dot(w)) != Y)

	def fit(self, X, Y):
		self.check(X, Y)

		if not (self.random_state is None):
			if self.shuffle == random.shuffle:
				tmp = list(zip(X, Y))
				random.seed(self.random_state)
				self.shuffle(tmp)
				X, Y = zip(*tmp) # return tuple 		
			else:
				X, Y = self.shuffle(X, Y, self.random_state)
			X = np.array(X)
			Y = np.array(Y)
		
		n_data = X.shape[0]
		n_features = X.shape[1]
		self.coef_ = np.zeros(n_features)
		w = np.zeros(n_features)

		if self.pocket == True:
			error_cur = self.count_error(X, Y, self.coef_)
			while (self.n_updates < self.max_iter) and (error_cur > 0):
				for i in range(n_data):
					if np.sign(X[i].dot(w) * Y[i]) <= 0:
						self.n_updates += 1
						w += self.eta0 * Y[i] * X[i]
						error_new = self.count_error(X, Y, w)
						if error_new < error_cur:
							error_cur = error_new
							self.coef_ = np.copy(w)
		else:
			error = True
			while (self.n_updates < self.max_iter) and (error == True):
				error = False
				for i in range(n_data):
					if np.sign(X[i].dot(self.coef_) * Y[i]) <= 0:
						self.n_updates += 1
						self.coef_ += self.eta0 * Y[i] * X[i]
						error = True
			'''
			i = 0
			while (self.n_updates < self.max_iter) and (self.count_error(X, Y, self.coef_)):
				if np.sign(X[i].dot(self.coef_) * Y[i]) <= 0:
						self.n_updates += 1
						self.coef_ += self.eta0 * Y[i] * X[i]
				i += 1
				if i == n_data:
					i = 0
			'''
	def predict(self, X):
		self.check(X)

		n_data = len(X)
		predicted_list = [np.sign(X[i].dot(self.coef_)) for i in range(n_data)]
		return predicted_list

	def score(self, X, Y):
		self.check(X)
		pred = self.predict(X)
		if len(Y) != len(pred):
			raise ValueError('Shape of X does not match Y')
		else:
			Y = np.array(Y)
			pred = np.array(pred)
			return (Y == pred).mean()