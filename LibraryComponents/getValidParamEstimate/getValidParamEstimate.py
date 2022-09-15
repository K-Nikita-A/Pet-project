import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import numpy as np


def getValidParamEstimate(model: BaseEstimator, X: pd.DataFrame, y: pd.DataFrame | list,
                          param_name: str, param_range: range, scoring_name: str = 'neg_log_loss'):
    """
    Метод для получения графика влияния гиперпараметра на оценку обучения и оценку валидации
    :param model: Исследуемая модель
    :param X: Данные фичей
    :param y: Данные таргета
    :param param_name: Название исследуемого параметра
    :param param_range: Диапазон значений параметра
    :param scoring_name: Метрика, по которой исследуется модель
    :return: график
    """
    train_scores, valid_scores = validation_curve(model,
                                                  X=X, y=y,
                                                  param_name=param_name,
                                                  scoring=scoring_name,
                                                  param_range=param_range,
                                                  cv=5)

    train_mean = np.mean(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)

    train_std = np.std(train_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)

    plt.subplots(1, figsize=(7, 6))
    plt.plot(param_range, train_mean, label="Training score", color="purple")
    plt.plot(param_range, valid_mean, label="Cross-validation score", color="green")

    plt.fill_between(param_range, train_mean - train_std, train_mean + valid_std, color="red")
    plt.fill_between(param_range, valid_mean - valid_std, valid_mean + valid_std, color="gainsboro")

    plt.title("Validation Curve With " + type(model).__name__)
    plt.xlabel("Number Of " + param_name)
    plt.ylabel(scoring_name)
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show()

    return 0
