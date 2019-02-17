import os
import math
import argparse
import dbh_util as util

from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser()
parser.add_argument('--penalty', default='l2', help='Used to specify the norm used in the penalization, "l1" or "l2"') # l1 only for liblinear or saga
parser.add_argument('--solver', default='liblinear', help='Solver algorithm, "newton-cg", "lbfgs", "liblinear", "sag", or "saga"')
parser.add_argument('--C', type=float, default=1.0, help='Inverse of regularization strength')
parser.add_argument('--tol', type=float, default=1e-4, help='Tolerance for stopping criteria')

#
# Logistic reggression --> classification
# Regression because it predicts the probabilities of the possible classes...
#

def learn(train, dev, test, args, sargs_str):
    sargs = util.parse(parser, sargs_str.split())
    return util.sklearn_wrapper(train, dev, test, LogisticRegression(**sargs))