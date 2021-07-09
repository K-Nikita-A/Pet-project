from pandas import DataFrame, read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

data = DataFrame(read_csv('./resource/wine.data'))
className = data.iloc[:, 0]
features = scale(data.iloc[:, 1:14])

kf = KFold(n_splits=5, shuffle=True)

max_score = 0
max_k = 1
for i in range(49):
    clf = KNeighborsClassifier(n_neighbors=i + 1)
    score = round(cross_val_score(clf, features, className, cv=kf, scoring='accuracy').mean(), 2)
    if (max_score < score) | (max_score == 0):
        max_score = score
        max_k = i
print(max_score, max_k)
