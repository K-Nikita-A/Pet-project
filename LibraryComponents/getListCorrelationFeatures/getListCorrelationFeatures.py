import pandas as pd
import numpy as np
import warnings

warnings.simplefilter("ignore")


def getListCorrelationFeatures(df: pd.DataFrame, cut: float = 0.5) -> []:
    """
    Метод для получения списка факторов, которые коррелируют с другими фичами
    :param df: исходный датафрейм
    :param cut: порог, выше которого считаем факторы коррелируемыми
    :return: список коррелируемых факторов
    """
    corr_matrix = df.corr().abs()
    avg_corr = corr_matrix.mean(axis=1)
    up_corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    res = pd.DataFrame(columns=(['feature_1', 'feature_2', 'corr', 'feature_drop']))

    for row in range(len(up_corr_matrix) - 1):
        col_index = row + 1
        for col in range(col_index, len(up_corr_matrix)):
            if corr_matrix.iloc[row, col] > cut:
                drop = corr_matrix.columns[row] \
                    if avg_corr.iloc[row] > avg_corr.iloc[col] else corr_matrix.columns[col]
                s = pd.Series([corr_matrix.index[row],
                               up_corr_matrix.columns[col],
                               up_corr_matrix.iloc[row, col],
                               drop],
                              index=res.columns)

                res = res.append(s, ignore_index = True)

    all_corr_vars = list(set(res['feature_1'].tolist() + res['feature_2'].tolist()))
    poss_drop = list(set(res['feature_drop'].tolist()))

    diff = list(set(all_corr_vars).difference(set(poss_drop)))
    p = res[res['feature_1'].isin(diff) | res['feature_2'].isin(diff)][['feature_1', 'feature_2']]
    q = list(set(p['feature_1'].tolist() + p['feature_2'].tolist()))

    drop = (list(set(q).difference(set(diff))))
    poss_drop = list(set(poss_drop).difference(set(drop)))
    m = res[res['feature_1'].isin(poss_drop) | res['feature_2'].isin(poss_drop)][['feature_1',
                                                                                  'feature_2',
                                                                                  'feature_drop']]

    more_drop = set(list(m[~m['feature_1'].isin(drop) & ~m['feature_2'].isin(drop)]['feature_drop']))
    return [*drop, *more_drop]


