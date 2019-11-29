import os
import math
import dbh_util as util

from sklearn.naive_bayes import GaussianNB

def predict(classifier, test, args, sargs_str, threshold=None):
    preds = classifier.predict(test[0])
    if threshold is not None:
        preds = [1 if x >= threshold else 0 for x in preds]
    return preds

def learn(train, dev, test, args, sargs_str):
    return util.sklearn_wrapper(train, dev, test, GaussianNB())