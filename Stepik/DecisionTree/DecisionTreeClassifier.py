from pandas import read_csv, DataFrame
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV


def isNaN(num):
    return num != num


data = read_csv('./resources/train.csv')

df = DataFrame([])
df['Pclass'] = data['Pclass']
df['Fare'] = data['Fare']
df['Age'] = data['Age']
df['Sex'] = data['Sex']

df.loc[df['Sex'] == 'male', 'Sex'] = 1
df.loc[df['Sex'] == 'female', 'Sex'] = 0
df = df.drop(df.loc[isNaN(df['Age']), 'Age'].index, axis=0)
clf = tree.DecisionTreeClassifier()
clf.fit(df.iloc[:, 1:4].to_numpy(), df['Pclass'].to_numpy())
x_train, x_test, y_train, y_test = train_test_split(
    df.iloc[:, 1:4].to_numpy(), df['Pclass'].to_numpy(), test_size=0.33)

parameters = {'criterion': ['gini', 'entropy'], 'max_depth': range(1, 30)}
kf = KFold(n_splits=5, shuffle=True)
grid_search_cv_clf = GridSearchCV(clf, parameters, cv=kf)

grid_search_cv_clf.fit(x_train, y_train)
best_params = grid_search_cv_clf.best_params_
best_clf = grid_search_cv_clf.best_estimator_

y_pred = best_clf.predict(x_test)
score = best_clf.score(x_test, y_test)
accuracy = accuracy_score(y_test, y_pred)
precision_score = precision_score(y_test, y_pred, average='macro')
recall_score = recall_score(y_test, y_pred, average='macro')
y_predict_prob = best_clf.predict_proba(x_test)

print(clf.feature_importances_)
print(y_predict_prob)
print('best_clf: ', best_clf)
print('best_params: ', best_params)
print('score: ', score)
print('precision_score: ', precision_score)
print('recall_score: ', recall_score)
print('accuracy: ', accuracy)
