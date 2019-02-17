import os
import math
import dbh_util as util

from sklearn.naive_bayes import GaussianNB

def learn(train, dev, test, args, sargs_str):
    return util.sklearn_wrapper(train, dev, test, GaussianNB())