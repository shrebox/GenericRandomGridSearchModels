import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, matthews_corrcoef
import pandas as pd

# Configuration section
iter = 5
cvCount = 6
seed = 42
thresholdRange = np.linspace(start=0.46, stop=0.54, num=50)

# Load list of best parameters from Random Search
with open('ListOfBestParamsGS.pkl', 'rb') as f:
    best_params = pickle.load(f)


def getPredictionsGivenThreshold(myMatrix, th):
    myList = []
    for i in range(myMatrix.shape[0]):
        p1 = myMatrix[i, 1]
        if p1>= th:
            myList.append(1)
        else:
            myList.append(0)
    return np.asarray(myList)


thresholdList = []
precisionList = []
recallList = []
aucList = []
accuracyList = []
mcList = []
for threshold in thresholdRange:
    print(threshold)
    overallPrecision = 0
    overallRecall = 0
    overallAuauc = 0
    overallAccuracy = 0
    overallMc = 0
    for i in range(iter):
        X_train = np.load('X_test_' + str(i) + '.npy')
        Y_train = np.load('Y_test_' + str(i) + '.npy')
        skf = StratifiedKFold(n_splits=cvCount, random_state=seed)
        foldPrecision = 0
        foldRecall = 0
        foldAuauc = 0
        foldAccuracy = 0
        foldMc = 0
        for train_index, test_index in skf.split(X_train, Y_train):
            X_tr, X_te = X_train[train_index], X_train[test_index]
            Y_tr, Y_te = Y_train[train_index], Y_train[test_index]
            bp = best_params[i]
            clf = AdaBoostClassifier(base_estimator=bp['base_estimator'], n_estimators=bp['n_estimators'],
                                     algorithm=bp['algorithm'], random_state=seed).fit(X_tr, Y_tr.ravel())
            predictionsProb = clf.predict_proba(X_te)
            predictions = getPredictionsGivenThreshold(predictionsProb, threshold)
            precision = precision_score(Y_te, predictions)
            recall = recall_score(Y_te, predictions)
            fpr, tpr, thresholds = roc_curve(Y_te, predictions, pos_label=1)
            auroc = roc_auc_score(Y_te, predictionsProb[:, 1])
            accuracy = accuracy_score(Y_te, predictions)
            matthewsCoeff = matthews_corrcoef(Y_te, predictions)

            foldPrecision += precision
            foldRecall += recall
            foldAuauc += auroc
            foldAccuracy += accuracy
            foldMc += matthewsCoeff
        overallPrecision = overallPrecision + (foldPrecision/cvCount)
        overallRecall = overallRecall + (foldRecall/cvCount)
        overallAuauc = overallAuauc + (foldAuauc/cvCount)
        overallAccuracy = overallAccuracy + (foldAccuracy/cvCount)
        overallMc = overallMc + (foldMc/cvCount)
    thresholdList.append(threshold)
    precisionList.append(overallPrecision/iter)
    recallList.append(overallRecall/iter)
    aucList.append(overallAuauc/iter)
    accuracyList.append(overallAccuracy/iter)
    mcList.append(overallMc/iter)

df = pd.DataFrame()
df['Threshold'] = thresholdList
df['Precision'] = precisionList
df['Recall'] = recallList
df['AUROC'] = aucList
df['Accuracy'] = accuracyList
df['MC'] = mcList
df.to_csv('Thresholding.csv', index=False)
print('Done')
