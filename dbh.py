#
# Imports
#

import os
import sys
import logging
import argparse
import numpy as np
import tensorflow as tf

import dbh_util as util
import csv2pandas
import preprocess
import strategies


#
# Arguments
#

# custom "list of pairs" arg type
# pairs are strings separated by a colon
def pair(arg):
    return [x for x in arg.split(':')]

# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--csv', required=True, help='csv to read the data from')
parser.add_argument('--label', required=True, help='name of the label to predict')
parser.add_argument('--resample', choices=['up', 'down', 'none'], default='none', help='in which direction to balance the training data')
parser.add_argument('--resample-amount', default=100, type=int, help='how much to balance the training data towards uniformity')
parser.add_argument('--seed', default=1337, type=int, help='random seed for repeatability')
parser.add_argument('--preprocess', default=[], type=pair, nargs='+', help='how to preprocess the input, given in WHAT:HOW pairs, where WHAT={features, labels} and HOW={binarize, normalize, standardize}')
parser.add_argument('--strategy', required=True, type=pair, nargs='+', help='how to build models, given in STRATEGY:"params to the strategy" pairs, where STRATEGY={cdnnc}')
parser.add_argument('--output', default=os.path.abspath('output'), help='output dir to write model and logs to')
parser.add_argument('--clean', default=False, help='clean output dir before starting?', action='store_true')
parser.add_argument('--device', default='/device:CPU:0', help='device to run learning on (cpu/gpu)')

#
# Constants
#

EPS = 1e-8
FOLDS = 10


#
# Helpers
#

class ConfMatrix:
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def add(self, eval_result):
        self.tp += eval_result['tp']
        self.tn += eval_result['tn']
        self.fp += eval_result['fp']
        self.fn += eval_result['fn']

    def stats(self):
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + EPS)
        precision = self.tp / (self.tp + self.fp + EPS)
        recall = self.tp / (self.tp + self.fn + EPS)
        fmes = (2 * precision * recall) / (precision + recall + EPS)

        return {
            'tp': self.tp,
            'tn': self.tn,
            'fp': self.fp,
            'fn': self.fn,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'fmes': fmes
        }


#
# Main
#

def main(args):

    # Create output folder
    util.mkdir(args['output'], args['clean'])

    # Tensorflow logging
    tf.logging.set_verbosity(tf.logging.WARN)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    # Logging
    logger = logging.getLogger('DeepBugHunter')
    if not len(logger.handlers):
        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(args['output'], 'dbh.log'), mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

    logger.info('DBH started...')

    # Seeding global random states, just in case...
    tf.set_random_seed(args['seed'])
    # This is used for sklearn algorithms under the hood so we don't have to manually
    # set the random seed separately every time
    np.random.seed(args['seed'])

    # Load the whole input
    data = csv2pandas.load_data(args['csv'], args['label'], args['seed'])

    # Apply optional preprocessing
    for (what, how) in args['preprocess']:
        # TODO: use <what> and generalize preprocessors
        data = getattr(preprocess, how)(*data)

    table = []
    strategy_i = 0
    strategy_cnt = len(args['strategy'])
    for (strategy, sargs) in args['strategy']:

        strategy_i += 1
        logger.info('(%d/%d) Strategy "%s" started with args: <%s>', strategy_i, strategy_cnt, strategy, sargs)

        # Aggregate confusion matrices
        cv_train = ConfMatrix()
        cv_dev = ConfMatrix()
        cv_test = ConfMatrix()

        # For each fold
        fold_generator = preprocess.split(data, folds=FOLDS, seed=args['seed'])
        fold_i = 0
        for remainder, test in fold_generator():
            fold_i += 1

            # A single dev split
            # Not fully fair, but fairer...
            train, dev = next(preprocess.split(remainder, folds=FOLDS, seed=args['seed'])())
            
            # Resample the training set
            if args['resample'] is not 'none':
                train = preprocess.resample(*train, mode=args['resample'], amount=args['resample_amount'], seed=args['seed'])

            # Evalute according to the current strategy
            train_res, dev_res, test_res, _ = getattr(strategies, strategy).learn(train, dev, test, args, sargs)

            # Aggregate metrics for cross-validation F-Measure
            cv_train.add(train_res)
            cv_dev.add(dev_res)
            cv_test.add(test_res)

            logger.info('Fold %d/10 done', fold_i)

        train_stats = cv_train.stats()
        dev_stats = cv_dev.stats()
        test_stats = cv_test.stats()

        logger.info('%s[%s] results:', strategy, sargs)
        logger.info('train: %s', train_stats)
        logger.info('dev:   %s', dev_stats)
        logger.info('test:  %s', test_stats)

        table.append([
            args['resample'],
            args['resample_amount'],
            args['preprocess'],
            strategy,
            sargs,
            train_stats['fmes'],
            dev_stats['fmes'],
            test_stats['fmes'],
            train_stats,
            dev_stats,
            test_stats,
        ])


    with open(os.path.join(args['output'], 'dbh.csv'), 'a') as f:
        for line in table:
            f.write(';'.join([str(item) for item in line]) + '\n')

if __name__ == '__main__':
    main(util.parse(parser, sys.argv[1:]))