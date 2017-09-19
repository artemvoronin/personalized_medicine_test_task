import os
import sys
import pickle
import argparse
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

NUM_FOLDS = 8
NUM_ESTIMATORS = 101
DEPTH = 7
WEIGHT = None  #"balanced"


def get_score(pred, true):
    assert len(pred) == len(true)
    acc = 0.0
    for i in range(len(pred)):
        if pred[i] == true[i]:
            acc += 1
    acc /= len(pred)
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training by decision trees.")
    parser.add_argument("--train", type=str, help="Train data pickled", required=True, dest='train')
    parser.add_argument("--test", type=str, help="Test data pickled", required=True, dest='test')
    args = parser.parse_args()

    if not os.path.exists(args.train):
        print("Cannot find train data at {}".format(args.train))
        sys.exit()
    if not os.path.exists(args.test):
        print("Cannot find test data at {}".format(args.test))
        sys.exit()

    print("Read train data...")
    train_obj = pickle.load(open(args.train, 'rb'))
    train_x = train_obj['train_x']
    train_y = train_obj['train_y']

    print("Read test data...")
    test_obj = pickle.load(open(args.test, 'rb'))
    test_x = test_obj['test_x']
    test_y = test_obj['test_y']

    print("Training...")
    fold_scores = []
    kfold = KFold(n_splits=NUM_FOLDS, random_state=None, shuffle=True)
    for train_index, val_index in kfold.split(train_x):
        print("  Fold #{:d}".format(len(fold_scores)))
        train_x_fold, val_x_fold = train_x[train_index], train_x[val_index]
        train_y_fold, val_y_fold = train_y[train_index], train_y[val_index]

        classifier = RandomForestClassifier(
            n_estimators=NUM_ESTIMATORS,
            max_depth=DEPTH,
            oob_score=True,
            class_weight=WEIGHT,
            verbose=False)
        classifier.fit(train_x_fold, train_y_fold)

        train_y_hat = classifier.predict(train_x_fold)
        val_y_hat = classifier.predict(val_x_fold)

        train_score = get_score(train_y_hat, train_y_fold)
        test_score = get_score(val_y_hat, val_y_fold)
        fold_scores.append((train_score, test_score))

    print("Fold results...")
    print("Fold  Train-score  Test-val")
    for i in range(NUM_FOLDS):
        print("{:d}    {:7.4f}      {:7.4f} ".format(i, fold_scores[i][0], fold_scores[i][1]))

    print("Final classifier...")
    classifier = RandomForestClassifier(
        n_estimators=NUM_ESTIMATORS,
        max_depth=DEPTH,
        oob_score=True,
        class_weight=WEIGHT,
        verbose=False)
    classifier.fit(train_x, train_y)
    y_hat = classifier.predict_proba(test_x)

    print("Saving results for submission...")
    with open("submission_sp64.csv", "w") as fp:
        fp.write("ID,class1,class2,class3,class4,class5,class6,class7,class8,class9\n")
        for id in range(len(y_hat)):
            line = str(id)
            for j in range(9):
                line += ',' + str(y_hat[id, j])
            line += '\n'
            fp.write(line)

    # Loss function fucks up quantized submissions.
    #print("Saving results for submission...")
    #with open("submission_sp64.csv", "w") as fp:
    #    fp.write("ID,class1,class2,class3,class4,class5,class6,class7,class8,class9\n")
    #    for id in range(len(y_hat)):
    #        result = ['0'] * 9
    #        result[y_hat[id] - 1] = '1'
    #        line = str(id) + ',' + ','.join(result) + '\n'
    #        fp.write(line)

    print("Done.")
