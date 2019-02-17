import os
import math
import dbh_util as util

from sklearn.dummy import DummyClassifier

#
# ZeroR
#

def learn(train, dev, test, args, sargs_str):
    return util.sklearn_wrapper(train, dev, test, DummyClassifier(strategy='most_frequent'))