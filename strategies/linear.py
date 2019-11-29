import os
import math
import dbh_util as util

from sklearn.linear_model import LinearRegression

#
# Linear reggression
# Regression values binned as above or below .5 to make a classifier
#

def predict(classifier, test, args, sargs_str, threshold=.5):
    preds = classifier.predict(test[0])
    if threshold is not None:
        preds = [1 if x >= threshold else 0 for x in preds]
    return preds

def learn(train, dev, test, args, sargs_str):
    return util.sklearn_wrapper(train, dev, test, LinearRegression(), .5)