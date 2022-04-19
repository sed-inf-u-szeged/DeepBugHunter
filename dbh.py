#
# Imports
#

import os
import sys
import logging
import logging.handlers
import math
import argparse
import itertools
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
def _pair(arg):
    return [x for x in arg.split(':')]

# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--csv', required=True, help='csv to read the data from')
parser.add_argument('--label', required=True, help='name of the label to predict')
parser.add_argument('--resample', choices=['up', 'down', 'none'], default='none', help='in which direction to balance the training data')
parser.add_argument('--resample-amount', default=100, type=int, help='how much to balance the training data towards uniformity')
parser.add_argument('--seed', default=1337, type=int, help='random seed for repeatability')
parser.add_argument('--preprocess', default=[], type=_pair, nargs='+', help='how to preprocess the input, given in WHAT:HOW pairs, where WHAT={features, labels} and HOW={binarize, normalize, standardize}')
parser.add_argument('--strategy', required=True, type=_pair, nargs='+', help='how to build models, given in STRATEGY:"params to the strategy" pairs, where STRATEGY={cdnnc}')
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
        self.history = list()
        self.issue_history = list()
        self.covered_issues = 0
        self.missed_issues = 0

    def add(self, eval_result):
        self.history.append(eval_result)
        self.tp += eval_result['tp']
        self.tn += eval_result['tn']
        self.fp += eval_result['fp']
        self.fn += eval_result['fn']

    def calc_completeness(self, preds, issues):
        pred_issues = zip(preds, issues)
        covered_issues = sum(map(lambda x: x[1] if x[0] == 1 else 0, pred_issues))
        self.covered_issues += covered_issues
        pred_issues = zip(preds, issues)
        missed_issues = sum(map(lambda x: x[1] if x[0] == 0 else 0, pred_issues))
        self.missed_issues += missed_issues
        self.issue_history.append((covered_issues, missed_issues))
    
    def stats(self, compl=False):
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + EPS)
        precision = self.tp / (self.tp + self.fp + EPS)
        recall = self.tp / (self.tp + self.fn + EPS)
        fmes = (2 * precision * recall) / (precision + recall + EPS)
        mcc = (float(self.tp)/1000.0 * float(self.tn)/1000.0 - float(self.fp)/1000.0 * float(self.fn)/1000.0) / (math.sqrt((float(self.tp)/1000.0 + float(self.fp)/1000.0)*(float(self.tp)/1000.0 + float(self.fn)/1000.0)*(float(self.tn)/1000.0 + float(self.fp)/1000.0)*(float(self.tn)/1000.0 + float(self.fn)/1000.0)) + EPS)

        ret = {
            'tp': self.tp,
            'tn': self.tn,
            'fp': self.fp,
            'fn': self.fn,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'fmes': fmes,
            'mcc': mcc
        }
        
        if compl:
            ret.update({
                'covered_issues': self.covered_issues,
                'missed_issues': self.missed_issues,
                'completeness': float(self.covered_issues)/(self.covered_issues+self.missed_issues) if self.covered_issues+self.missed_issues != 0 else 'NaN'
            })
            
        ret.update({
            'std_dev': self._calc_devs(compl)
        })
        
        return ret
        
    def _calc_devs(self, compl=False):
        data_frame = list()
        for eval_result, issue_result in itertools.zip_longest(self.history, self.issue_history):
            tp =  eval_result['tp']
            tn = eval_result['tn']
            fp = eval_result['fp']
            fn = eval_result['fn']
            accuracy = (tp + tn) / (tp + tn + fp + fn + EPS)
            precision = tp / (tp + fp + EPS)
            recall = tp / (tp + fn + EPS)
            fmes = (2 * precision * recall) / (precision + recall + EPS)
            mcc = (tp * tn - fp * fn) / (math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)) + EPS)
            data_frame.append([tp, tn, fp, fn, accuracy, precision, recall, fmes, mcc])
            if compl:
                c_issues = issue_result[0]
                m_issues = issue_result[1]
                completeness = float(c_issues)/(c_issues + m_issues) if c_issues + m_issues != 0 else 0
                data_frame[-1].extend([c_issues, m_issues, completeness])
        return np.std(data_frame, axis=0).tolist()
        
#
# Main
#

def main(args):
    # Create output folder
    util.mkdir(args['output'], args['clean'])

    # Tensorflow logging
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    # Logging to DWF server
    dwf_logging = None

    # Logging
    logger = logging.getLogger('DeepBugHunter')
    
    if 'dwf_client_info' in args:
        client_info = args['dwf_client_info']
        sys.path.insert(0, client_info['util_path'])
        dwf_logging = __import__('dwf_logging')        
            
    if not logger.handlers:
        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(args['output'], 'dbh.log'), mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        if 'dwf_client_info' in args:
            http_handler = dwf_logging.LogHandler()
            http_handler.setLevel(logging.INFO)
            logger.addHandler(http_handler)
        else:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

    extra_log_data = {}
    if dwf_logging is not None:
        extra_log_data = {'progress' : 0, 'hash' : client_info['client_id']}

    logger.info(msg='DBH started...', extra=extra_log_data)
    logger.info('Input csv is ' + args['csv'])

    # Seeding global random states, just in case...
    tf.compat.v1.set_random_seed(args['seed'])
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
            train_res, dev_res, test_res, cl = getattr(strategies, strategy).learn(train, dev, test, args, sargs)
            
            # Aggregate metrics for cross-validation F-Measure
            cv_train.add(train_res)
            cv_dev.add(dev_res)
            cv_test.add(test_res)
            
            if args['calc_completeness']:
                preds = getattr(strategies, strategy).predict(cl, dev, args, sargs)
                issues = preprocess.get_orig_labels(dev[1])
                cv_dev.calc_completeness(preds, issues)
                
                preds = getattr(strategies, strategy).predict(cl, test, args, sargs)
                issues = preprocess.get_orig_labels(test[1])
                cv_test.calc_completeness(preds, issues)


            if dwf_logging is not None:
                extra_log_data = {'progress' : fold_i / FOLDS, 'hash' : client_info['client_id']}
            
            logger.info('Fold %d/10 done', fold_i, extra=extra_log_data)

        train_stats = cv_train.stats(False)
        dev_stats = cv_dev.stats(args['calc_completeness'])
        test_stats = cv_test.stats(args['calc_completeness'])

        logger.info('%s[%s] results:', strategy, sargs)
        logger.info('train: %s', train_stats)
        logger.info('dev:   %s', dev_stats)
        logger.info('test:  %s', test_stats)

        if dwf_logging is not None:
            result = dwf_logging.pack_results(train_stats, dev_stats, test_stats)
            dwf_logging.report_result(result, client_info['client_id'])            


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