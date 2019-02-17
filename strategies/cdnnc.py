import os
import math
import logging
import argparse
import tensorflow as tf
from tensorflow.python.platform import tf_logging

import dbh_util as util
import classifier as cl
import pandas2tf

EPS = 1e-8
CLASSES = 2

def log(msg):
    tf_logging.log(tf_logging.FATAL, msg) # FATAL to show up at any TF logging level
    logging.getLogger('DeepBugHunter').info(msg)

#
# Strategy args
#

parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=int, help='Number of layers')
parser.add_argument('--neurons', type=int, help='Number of neurons per layer')
parser.add_argument('--batch', type=int, help='Batch size')
parser.add_argument('--lr', type=float, help='Starting learning rate')
parser.add_argument('--beta', type=float, default=0.0, help='L2 regularization bias')
parser.add_argument('--max-misses', type=int, default=4, help='Maximum consecutive misses before early stopping')
parser.add_argument('--sandbox', default=os.path.abspath('sandbox'), help='Intermediary model folder')



#
# Validate after every epoch, and if the model gets worse, then restore the previous best model and try again
# with a reduced (halved) learning rate
#

def learn(train, dev, test, args, sargs_str):

    # Read strategy-specific args
    sargs = util.parse(parser, sargs_str.split())
   
    # Clean out the sandbox
    util.mkdir(sargs['sandbox'], clean=True)

    # Feature columns describe how to use the input
    my_feature_columns = []
    for key in train[0].keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Calculate epoch length
    steps_per_epoch = math.ceil(len(train[0]) / sargs['batch'])

    # Train a classifier
    # Repeat until the model consecutively "misses" a set number of times
    rounds = 1
    misses = miss_streak = 0
    best_result = {'fmes': -1}
    best_model_dir = None
    best_classifier = None
    while miss_streak < sargs['max_misses']:

        model_dir = os.path.join(sargs['sandbox'], 'run_' + str(rounds) + '_' + str(miss_streak))

        extra_args = {
            'classes': CLASSES,
            'columns': my_feature_columns,
            'steps_per_epoch': steps_per_epoch,
            'learning_rate': sargs['lr'] / (2 ** misses),
            'model_dir': model_dir,
            'warm_start_dir': best_model_dir
        }
        merged_args = {**args, **sargs, **extra_args}

        # Create a new classifier instance
        classifier = cl.create_classifier(merged_args)

        # Train the model for exactly 1 epoch
        classifier.train(
            input_fn=lambda:pandas2tf.train_input_fn(train, sargs['batch']),
            steps=steps_per_epoch)

        # Evaluate the model
        eval_result = classifier.evaluate(input_fn=lambda:pandas2tf.eval_input_fn(dev, sargs['batch']))
        log('Round ' + str(rounds) + '_' + str(miss_streak) + ', Fmes: ' + str(best_result['fmes']) + ' --> ' + str(eval_result['fmes']))
        if eval_result['fmes'] > best_result['fmes']:
            best_result = eval_result
            best_model_dir = model_dir
            best_classifier = classifier
            miss_streak = 0
            rounds += 1
            log('Improvement, go on...')
        else:
            miss_streak += 1
            misses += 1
            log('Miss #' + str(misses) + ', (streak = ' + str(miss_streak) + ')')

    final_result_train = best_classifier.evaluate(input_fn=lambda:pandas2tf.eval_input_fn(train, sargs['batch']))
    final_result_dev = best_classifier.evaluate(input_fn=lambda:pandas2tf.eval_input_fn(dev, sargs['batch']))
    final_result_test = best_classifier.evaluate(input_fn=lambda:pandas2tf.eval_input_fn(test, sargs['batch']))
    return final_result_train, final_result_dev, final_result_test, best_classifier
       