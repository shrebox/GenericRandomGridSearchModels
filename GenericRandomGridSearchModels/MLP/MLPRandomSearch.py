import numpy as np
import pickle
import random
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

# Configuration section
iter = 5
cvCount = 6
seed = 42
wdiff = 0.5
wtest = 0.5
numSamples = 10000

# List of best parameters
best_params = []

# Define a custom scoring mechanism
def myScoring(y_pred_train, y_true_train, y_pred_test, y_true_test):
    train_acc = accuracy_score(y_true_train, y_pred_train)
    test_acc = accuracy_score(y_true_test, y_pred_test)
    acc_diff = (train_acc - test_acc)*(-1)
    return wdiff*acc_diff + wtest*test_acc, train_acc, test_acc


# Random search over parameters
for i in range(iter):
    X_train = np.load('X_train_' + str(i) + '.npy')
    Y_train = np.load('Y_train_' + str(i) + '.npy')

    numLayers = []
    for j in range(100):
        numLayers.append(random.randint(2, 5))
    listOfHiddenLayers = []
    for n in numLayers:
        inner_list = []
        for k in range(n):
            inner_list.append(random.randint(100, 200))
        listOfHiddenLayers.append(tuple(inner_list))

    activation = ['logistic', 'tanh', 'relu']
    solver = ['lbfgs', 'sgd', 'adam']
    alpha = [x for x in np.linspace(0.0001, 1, num=1000)]
    max_iter = [5000]

    grid = {'hidden_layer_sizes': numLayers,
            'activation': activation,
            'solver': solver,
            'alpha': alpha,
            'max_iter': max_iter
           }

    print('Searching')
    randomCombinations = random.sample(list(ParameterGrid(grid)), numSamples)
    score_list = []
    combination_list = []
    train_acc_list = []
    test_acc_list = []
    tracker = 0
    for combination in randomCombinations:
        print(tracker)
        skf = StratifiedKFold(n_splits=cvCount, random_state=seed, shuffle=True)
        s = 0
        tr_acc = 0
        te_acc = 0
        for train_idx, test_idx in skf.split(X_train, Y_train):
            split_x_train, split_x_test = X_train[train_idx], X_train[test_idx]
            y_true_train, y_true_test = Y_train[train_idx], Y_train[test_idx]
            mlp = MLPClassifier(**combination)
            clf = mlp.fit(split_x_train, y_true_train.ravel())
            y_pred_train = clf.predict(split_x_train)
            y_pred_test = clf.predict(split_x_test)
            score, fold_train_acc, fold_test_acc = myScoring(y_pred_train, y_true_train, y_pred_test, y_true_test)
            s += score
            tr_acc += fold_train_acc
            te_acc += fold_test_acc
        combination_list.append(combination)
        score_list.append(s / cvCount)
        train_acc_list.append(tr_acc / cvCount)
        test_acc_list.append(te_acc / cvCount)
        tracker += 1
    req_idx = score_list.index(max(score_list))
    best_params.append(combination_list[req_idx])
    print(str(i) + '-' + str(train_acc_list[req_idx]) + '-' + str(test_acc_list[req_idx]))

with open('ListOfBestParamsRS.pkl', 'wb') as f:
    pickle.dump(best_params, f)

print('Done')
