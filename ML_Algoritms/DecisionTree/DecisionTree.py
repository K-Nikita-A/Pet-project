from pandas import read_csv, DataFrame
from sklearn.tree import DecisionTreeClassifier


def isNaN(num):
    return num != num


data = read_csv('resources/train.csv')
print(data.columns)

df = DataFrame([])
df['Pclass'] = data['Pclass']
df['Fare'] = data['Fare']
df['Age'] = data['Age']
df['Sex'] = data['Sex']

df.loc[df['Sex'] == 'male', 'Sex'] = 1
df.loc[df['Sex'] == 'female', 'Sex'] = 0
df = df.drop(df.loc[isNaN(df['Age']), 'Age'].index, axis=0)

clf = DecisionTreeClassifier(random_state=241)
clf.fit(df.iloc[:, 1:4].to_numpy(), df['Pclass'].to_numpy())
print(clf.feature_importances_)
