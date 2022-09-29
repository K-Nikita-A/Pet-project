import pandas as pd
import xgboost as xgb
from LibraryComponents.showValidParamEstimate import showValidParamEstimate


def main():
    data = pd.read_csv('../data/train.csv')
    y = data.target
    X = data.drop('target', axis=1)
    showValidParamEstimate(xgb.XGBClassifier(), X, y, 'n_estimators', range(0, 30), 'neg_log_loss')
    return 0


if __name__ == "__main__":
    main()
