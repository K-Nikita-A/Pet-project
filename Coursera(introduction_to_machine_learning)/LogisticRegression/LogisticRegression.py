from typing import Tuple

from numpy import *
from pandas import read_csv
from sklearn.metrics import roc_auc_score

data = read_csv('resources/data-logistic.csv', header=None)
y = data[0]
x = data.loc[:, 1:2]

k = 0.1
l = len(y)


def calc_w1(w1_old: float, w2_old: float, C: float) -> float:
    sum = 0
    for i in range(0, l):
        expression_exp = -y[i] * (w1_old * x[1][i] + w2_old * x[2][i])
        sum = sum + y[i] * x[1][i] * (1 - 1 / (1 + exp(expression_exp)))
    return w1_old + k * sum / l - k * C * w1_old


def calc_w2(w1_old: float, w2_old: float, C: float) -> float:
    sum = 0
    for i in range(0, l):
        expression_exp = -y[i] * (w1_old * x[1][i] + w2_old * x[2][i])
        sum = sum + y[i] * x[2][i] * (1 - 1 / (1 + exp(expression_exp)))
    return w2_old + k * sum / l - k * C * w2_old


epsilon = 1e-5
max_steps = 10000


def grad_boost(C: float) -> Tuple[float, float]:
    w1_prew = 0.0
    w2_prew = 0.0
    w1 = 0.0
    w2 = 0.0
    for i in range(max_steps):
        w1 = calc_w1(w1_prew, w2_prew, C)
        w2 = calc_w2(w1_prew, w2_prew, C)
        if sqrt((w1 - w1_prew) ** 2 + (w2 - w2_prew) ** 2) <= epsilon:
            break
        else:
            w1_prew = w1
            w2_prew = w2
    return w1, w2


w1_with_reg, w2_with_reg = grad_boost(10.0)
w1_without_reg, w2_without_reg = grad_boost(0.0)


def sigmoid(w1: float, w2: float):
    return 1 / (1 + exp(-w1 * x[1] - w2 * x[2]))


y_with_reg = sigmoid(w1_with_reg, w2_with_reg)
y_without_reg = sigmoid(w1_without_reg, w2_without_reg)

auc_with_reg = roc_auc_score(y, y_with_reg)
auc_without_reg = roc_auc_score(y, y_without_reg)

print(round(auc_with_reg, 3), round(auc_without_reg, 3))
