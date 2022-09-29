from pandas import read_csv
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

data_train = read_csv('resources/perceptron-train.csv')
data_test = read_csv('resources/perceptron-test.csv')

y_train = data_train['target']
x_train = data_train.drop(['target'], axis=1)
y_test = data_test['target']
x_test = data_test.drop(['target'], axis=1)

clf = Perceptron(max_iter=5, tol=None, random_state=241)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
clf.fit(x_train_scaled, y_train)
y_pred_scaled = clf.predict(x_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
print(accuracy, accuracy_scaled, round(accuracy_scaled - accuracy, 3))