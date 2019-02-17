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

def learn(train, dev, test, args, sargs_str):
    sargs = util.parse(parser, sargs_str.split())
    return util.sklearn_wrapper(train, dev, test, RandomForestClassifier(**sargs))
