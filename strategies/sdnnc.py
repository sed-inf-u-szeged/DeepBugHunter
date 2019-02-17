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
MAX_MISSES = 4

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
parser.add_argument('--epochs', type=int, help='Epoch count')
parser.add_argument('--lr', type=float, help='Starting learning rate')
parser.add_argument('--beta', type=float, default=0.0, help='L2 regularization bias')
parser.add_argument('--sandbox', default=os.path.abspath('sandbox'), help='Intermediary model folder')



#
# Simple DNN classification for a set number of epochs
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
    total_steps = sargs['epochs'] * steps_per_epoch

    # Train a classifier
    extra_args = {
        'classes': CLASSES,
        'columns': my_feature_columns,
        'steps_per_epoch': steps_per_epoch,
        'learning_rate': sargs['lr'],
        'model_dir': sargs['sandbox'],
        'warm_start_dir': None
    }
    merged_args = {**args, **sargs, **extra_args}

    # Create a new classifier instance
    classifier = cl.create_classifier(merged_args)

    # Train the model for exactly 1 epoch
    classifier.train(
        input_fn=lambda:pandas2tf.train_input_fn(train, sargs['batch']),
        steps=total_steps)

    # Evaluate the model
    train_result = classifier.evaluate(input_fn=lambda:pandas2tf.eval_input_fn(train, sargs['batch']))
    dev_result = classifier.evaluate(input_fn=lambda:pandas2tf.eval_input_fn(dev, sargs['batch']))
    test_result = classifier.evaluate(input_fn=lambda:pandas2tf.eval_input_fn(test, sargs['batch']))
    return train_result, dev_result, test_result, classifier
       