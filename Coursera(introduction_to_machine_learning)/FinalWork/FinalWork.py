import numpy as np
from datetime import datetime
from pandas import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

### Подход 1: градиентный бустинг "в лоб"

print("Gradient boosting classifier: ")
data = read_csv('resources/features.csv')
data = data.drop(
    ["duration", "tower_status_radiant", "tower_status_dire", "barracks_status_radiant", "barracks_status_dire"],
    axis=1)

for col in DataFrame(data).columns:
    if len(data) - data[col].count() > 0:
        print(col)

# Ответ
# first_blood_time - время первой крови - первой крови нет в первые 5 минут игры.
# first_blood_team - команда, совершившая первой ранение - также как first_blood_time, в первые 5 мин не часто убивают.

data = data.fillna(0)

y_train = data["radiant_win"]
x_train = data.drop("radiant_win", axis=1)

# Ответ - radiant_win

kf = KFold(n_splits=5, shuffle=True, random_state=42)

n_estimators = [1,10,20,25,30,35,40,45]
for n_estimator in n_estimators:
    clf = GradientBoostingClassifier(n_estimators=n_estimator, random_state=42)
    start_time = datetime.now()
    score = cross_val_score(clf, x_train, y_train, cv=kf, scoring="roc_auc", n_jobs=-1).mean()
    print("n_estimator: ", n_estimator)
    print("score: ", score)
    print("time: ", datetime.now() - start_time)

# Ответ
# 1. Кросс-валидация для градиентного бустинга с 30
# деревьями проводилась 0:00:22 и показатель метрики качества равен 0.69
# 2. При увелечении числа деревьев показатель скоринга увеличивается до 0.7,
# но также увеличивается время работы, поэтому можно использовать большее колличество деревьев
# Чтобы ускорить работу алгоритма можно отфильтровать часть данных, убрав выбросы. А также поиграть с параметрами классификатора.

## Подход 2: логистическая регрессия

print('Logistic regression: ')
list_C = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]


def getLogisticScore(x_train):
    scaler = StandardScaler()
    for c in list_C:
        clf = LogisticRegression(C=c, random_state=42)
        start_time = datetime.now()
        score = cross_val_score(
            clf,
            DataFrame(scaler.fit_transform(x_train), index=x_train.index, columns=x_train.columns),
            y_train, cv=kf, scoring='roc_auc', n_jobs=-1).mean()
        print("c: ", c)
        print("score: ", score)
        print("time: ", datetime.now() - start_time)


getLogisticScore(x_train)

# Ответ
# Наилучший параметр регуляризации с равен 0.0001. Скоринг при нём равен 0.72. Время 0:00:01.99. Работае заметно быстрее.
# Градиентный бустинг даёт такой результат на колличестве n_estimators большем 100 (не проверял больше 45)
# Думаю, разница в том, что зависимость в данных близка к линейной, поэтому линейной регресси достаточно, чтобы хорошо её найти.

list_columns = ['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero',
                'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero',
                'd4_hero', 'd5_hero']

x_train_filter = x_train.drop(list_columns, axis=1)

getLogisticScore(x_train_filter)

# Ответ
# После удаления ряда признаков получился лучший результат также при С=0.0001 равный 0.7114122313361925
# До этого был 0.7114288784312144. Это значит, что модель никак не изменила свой результат, так как изначально
# эти признаки не рассматривались как влияющие на зависимость целевой переменной

print('Число различных идентификаторов героев: ',DataFrame(x_train.loc[:, list_columns]).nunique()['r1_hero'])

# Ответ
# Число различных идентификаторов героев:  108

N = DataFrame(x_train.loc[:, list_columns]).groupby('r1_hero', as_index=False).count()['r1_hero'].max()


def get_data_pick(data: DataFrame):
    X_pick = np.zeros((data.shape[0], N))

    for i, match_id in enumerate(data.index):
        for p in range(1, 6):
            X_pick[i, data.loc[match_id, f"r{p}_hero"] - 1] = 1
            X_pick[i, data.loc[match_id, f"d{p}_hero"] - 1] = -1

    return DataFrame(X_pick, index=data.index, columns=[f"new_{i}" for i in range(N)])


x_train_modify = concat([x_train, get_data_pick(data)], axis=1)

getLogisticScore(x_train_modify)

# Ответ
# После добавления "мешка слов" скоринг увеичился до 0.75 при c = 0.001.
# Думаю, это произошло по причине добавления новый признаков
