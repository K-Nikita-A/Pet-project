import pandas as pd
import xgboost as xgb
from LibraryComponents.getValidParamEstimate import getValidParamEstimate


def main():
    data = pd.read_csv('../data/train.csv')
    y = data.target
    X = data.drop('target', axis=1)
    getValidParamEstimate(xgb.XGBClassifier(), X, y, 'n_estimators', range(0, 30), 'neg_log_loss')
    return 0


if __name__ == "__main__":
    main()
