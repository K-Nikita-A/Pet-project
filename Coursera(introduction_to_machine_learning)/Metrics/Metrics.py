from pandas import read_csv
from sklearn.metrics import roc_auc_score

data = read_csv('resources/classification.csv')
predict = data['pred']
result = data['true']
TP = 0
FP = 0
FN = 0
TN = 0
for i in range(len(result)):
    if (result[i] == 1) & (predict[i] == 1):
        TP += 1
    else:
        if (result[i] == 0) & (predict[i] == 1):
            FP += 1
        else:
            if (result[i] == 0) & (predict[i] == 0):
                TN += 1
            else:
                if (result[i] == 1) & (predict[i] == 0):
                    FN += 1
print(TP, FP, FN, TN)
acc = round((TP + TN) / (TP + FP + FN + TN), 2)
pr = round(TP / (TP + FP), 2)
rec = round(TP / (TP + FN), 2)
f1 = round(2 * (pr ** -1 + rec ** -1) ** -1, 2)
print(acc, pr, rec, f1)

data_new = read_csv('resources/scores.csv')
auc_score_logreg = roc_auc_score(data_new['true'], data_new['score_logreg'])
auc_score_svm = roc_auc_score(data_new['true'], data_new['score_svm'])
auc_score_knn = roc_auc_score(data_new['true'], data_new['score_knn'])
auc_score_tree = roc_auc_score(data_new['true'], data_new['score_tree'])