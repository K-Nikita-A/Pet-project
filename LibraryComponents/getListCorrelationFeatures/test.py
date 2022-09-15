import pandas as pd
from getListCorrelationFeatures import getListCorrelationFeatures


def main():
    data = pd.read_csv('../data/train.csv')
    result_list = getListCorrelationFeatures(data.drop('target', axis=1))
    print(result_list)
    return 0


if __name__ == "__main__":
    main()
