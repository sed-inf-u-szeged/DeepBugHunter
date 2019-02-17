import os
import math
import dbh_util as util

from sklearn.linear_model import LinearRegression

#
# Linear reggression
# Regression values binned as above or below .5 to make a classifier
#

def learn(train, dev, test, args, sargs_str):
    return util.sklearn_wrapper(train, dev, test, LinearRegression(), .5)