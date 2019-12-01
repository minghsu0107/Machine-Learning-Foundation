import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from my_model.prepocessing import load
from sklearn.linear_model import Perceptron

X, Y = load('my_data/pocket_train.dat')
test_X, test_Y = load('my_data/pocket_test.dat')

errors = []
for i in range(1126):
	clf = Perceptron(max_iter=100, random_state=i)
	clf.fit(X, Y)
	errors.append(1 - clf.score(test_X, test_Y))
	print("training round {}".format(i + 1))

plt.figure()
ax = plt.axes()
ax.hist(errors, facecolor='green', alpha=0.8)
ax.set_xlabel('Error rate')
ax.set_ylabel('Frequency')
ax.set_title('Average error rate is {}'.format(np.array(errors).mean()))
plt.show()
