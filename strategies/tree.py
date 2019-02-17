import os
import math
import argparse
import dbh_util as util

from sklearn.tree import DecisionTreeClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--max-depth', type=int, default=5, help='Max decision tree leaf node depth')
parser.add_argument('--criterion', default='gini', help='Split quality criterion, "gini" or "entropy"')

#
# Decision Tree approach
#

def learn(train, dev, test, args, sargs_str):
    sargs = util.parse(parser, sargs_str.split())
    return util.sklearn_wrapper(train, dev, test, DecisionTreeClassifier(**sargs))