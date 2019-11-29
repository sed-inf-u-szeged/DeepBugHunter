import os
import math
import argparse
import dbh_util as util

from sklearn.neighbors import KNeighborsClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--n_neighbors', type=int, default=5, help='Number of neighbors')
parser.add_argument('--weights', default='uniform', help='Weighting method')

#
# K Nearest Neighbors approach
#

def predict(classifier, test, args, sargs_str, threshold=None):
    sargs = util.parse(parser, sargs_str.split())
    preds = classifier.predict(test[0])
    if threshold is not None:
        preds = [1 if x >= threshold else 0 for x in preds]
    return preds

def learn(train, dev, test, args, sargs_str):
    sargs = util.parse(parser, sargs_str.split())
    return util.sklearn_wrapper(train, dev, test, KNeighborsClassifier(**sargs))
