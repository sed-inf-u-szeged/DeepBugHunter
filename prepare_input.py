import sys
import math
import logging
import argparse
import pandas as pd
from sklearn.utils import shuffle
from sklearn.utils import resample as resamp
from sklearn.model_selection import StratifiedKFold

import dbh_util as util

parser = argparse.ArgumentParser()
parser.add_argument('--csv', required=True, help='csv to read the data from')
parser.add_argument('--label', required=True, help='name of the label to predict')
parser.add_argument('--seed', default=1337, type=int, help='random seed for repeatability')
parser.add_argument('--amount', type=float, default=100, help='percentage of the input to keep')
parser.add_argument('--output', required=True, help='csv to to save to')

args = util.parse(parser, sys.argv[1:])

# read full dataset
data = pd.read_csv(args['csv'], header=0)
print('Before resampling:\n%s', data[args['label']].value_counts())

# split to classes
bins = []
bins.append( data[data[args['label']] == 0] )
bins.append( data[data[args['label']] != 0] )

# resample ALL classes
for i, item in enumerate(bins):
    current = len(item)
    target = int(current * (args['amount'] / 100))
    bins[i] = resamp(bins[i], replace=False, n_samples=target, random_state=args['seed'])

# recombine classes to the whole dataset (and reshuffle!)
data = pd.concat(bins, ignore_index=True)
data = shuffle(data, random_state=args['seed'])

# output resampled data
print('After resampling:\n%s', data[args['label']].value_counts())
data.to_csv(args['output'], index=False)