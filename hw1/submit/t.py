from my_model.prepocessing import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

X, Y = load('my_data/pocket_train.dat')
test_X, test_Y = load('my_data/pocket_test.dat')

param_test = {'n_estimators':range(10, 20), 
				'max_depth':range(10, 20, 2), 
				'min_samples_split':range(2, 10, 5)}
gsearch = GridSearchCV(estimator = RandomForestClassifier(criterion='entropy',
						random_state=10, oob_score=True),
						param_grid = param_test, scoring='roc_auc',iid=False, cv=5)
 
gsearch.fit(X, Y)
clf = RandomForestClassifier(criterion='entropy', random_state=10, oob_score=True)
clf.set_params(**gsearch.best_params_)
clf.fit(X, Y)
print(clf.score(test_X, test_Y))