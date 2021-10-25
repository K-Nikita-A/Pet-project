from pandas import read_csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

data = read_csv('./resources/abalone.csv')
y = data['Rings']
x = data.drop('Rings', axis=1)
x['Sex'] = x['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

kf = KFold(n_splits=5, shuffle=True, random_state=1)
for i in range(1, 51):
    clf = RandomForestRegressor(random_state=1, n_estimators=i, n_jobs=-1)
    score = cross_val_score(clf, x, y, cv=kf, scoring='r2').mean()
    if score >= 0.52:
        print(i)
        break
