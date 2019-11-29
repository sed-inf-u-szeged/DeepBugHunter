import os
import math
import argparse
import dbh_util as util

from sklearn.ensemble import RandomForestClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--n-estimators', type=int, default=10, help='The number of trees in the forest')
parser.add_argument('--max-depth', type=int, default=5, help='Max decision tree leaf node depth')
parser.add_argument('--criterion', default='gini', help='Split quality criterion, "gini" or "entropy"')

#
# Random Forest approach
#

def predict(classifier, test, args, sargs_str, threshold=None):
    sargs = util.parse(parser, sargs_str.split())
    preds = classifier.predict(test[0])
    if threshold is not None:
        preds = [1 if x >= threshold else 0 for x in preds]
    return preds

def learn(train, dev, test, args, sargs_str):
    sargs = util.parse(parser, sargs_str.split())
    return util.sklearn_wrapper(train, dev, test, RandomForestClassifier(**sargs))
