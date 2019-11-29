import os
import math
import argparse
import dbh_util as util

from sklearn.svm import SVC

def float_or_string(choices):
    def checker(value):
        if value in choices:
            return value
        return float(value)
    return checker

parser = argparse.ArgumentParser()
parser.add_argument('--C', type=float, default=1.0, help='Error term weight')
parser.add_argument('--kernel', default='linear', help='SVM kernel, "linear", "poly", "sigmoid" or "rbf"')
parser.add_argument('--degree', type=int, default=3, help='Polynom degree for the "poly" kernel')
parser.add_argument('--gamma', type=float_or_string(['auto']), default='auto', help='Kernel coefficient for "rbf", "poly" and "sigmoid"')
parser.add_argument('--coef0', type=float, default=.0, help='Independent term in kernel function for "poly" and "sigmoid"')

#
# Support Vector Machine approach
#

def predict(classifier, test, args, sargs_str, threshold=None):
    sargs = util.parse(parser, sargs_str.split())
    preds = classifier.predict(test[0])
    if threshold is not None:
        preds = [1 if x >= threshold else 0 for x in preds]
    return preds

def learn(train, dev, test, args, sargs_str):
    sargs = util.parse(parser, sargs_str.split())
    return util.sklearn_wrapper(train, dev, test, SVC(**sargs))