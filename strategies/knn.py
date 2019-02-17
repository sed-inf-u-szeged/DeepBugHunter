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

def learn(train, dev, test, args, sargs_str):
    sargs = util.parse(parser, sargs_str.split())
    return util.sklearn_wrapper(train, dev, test, KNeighborsClassifier(**sargs))
