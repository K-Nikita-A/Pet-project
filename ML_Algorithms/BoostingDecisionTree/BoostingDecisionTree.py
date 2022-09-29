from matplotlib.pyplot import plot, show
from pandas import read_csv
from numpy import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from typing import Callable

data = read_csv('resources/gbm-data.csv')
y = data['Activity'].values
x = data.drop('Activity', axis=1).values


sigmoid: Callable[[float], float] = lambda y_pred: 1 / (1 + exp(-y_pred))


def results(model, X: array, y: array):
    return [log_loss(y, sigmoid(y_pred)) for y_pred in model.staged_decision_function(X)]


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=241)
learning_rate_list = [1, 0.5, 0.3, 0.2, 0.1]
min_results = {}
for i in learning_rate_list:
    clf = GradientBoostingClassifier(n_estimators=250,
                                     verbose=True,
                                     random_state=241,
                                     learning_rate=i)
    clf.fit(X_train, y_train)

    train_loss = results(clf, X_train, y_train)
    test_loss = results(clf, X_test, y_test)
    plot(test_loss, "r", linewidth=2)
    plot(train_loss, "g", linewidth=2)
    show()

    min_value = min(test_loss)
    min_index = test_loss.index(min_value) + 1
    min_results[i] = min_value, min_index