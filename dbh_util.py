import os
import stat
import shutil
import copy
import datetime
import numpy as np

import strategies
from sklearn.metrics import confusion_matrix

def mkdir(dir_name, clean=False):
    if clean:
        
        # Clear the TF cache to avoid open file handlers blocking
        # https://github.com/tensorflow/tensorflow/issues/9571
        from tensorflow.python.summary.writer import writer_cache
        writer_cache.FileWriterCache.clear()
        
        try:
            shutil.rmtree(dir_name)
        except OSError as e:
            print('Could not remove dir: ' + dir_name)
            raise
    try:
        os.makedirs(dir_name)
    except OSError as e:
        print('Could not create dir: ' + dir_name)

def parse(parser, args):
    return dict(vars(parser.parse_args(args)))

def conf_matrix_convert(conf):
    return {
        'tp': conf[1][1],
        'tn': conf[0][0],
        'fp': conf[0][1],
        'fn': conf[1][0]
    }

def sklearn_eval(classifier, data, threshold=None):
    preds = classifier.predict(data[0])
    if threshold is not None:
        preds = [1 if x >= threshold else 0 for x in preds]
    return conf_matrix_convert(confusion_matrix(data[1], preds))

def sklearn_wrapper(train, dev, test, alg, threshold=None):
    classifier = alg.fit(train[0], train[1])
    train_res = sklearn_eval(classifier, train, threshold)
    dev_res = sklearn_eval(classifier, dev, threshold)
    test_res = sklearn_eval(classifier, test, threshold)
    return train_res, dev_res, test_res, classifier

def _numpy_to_pytype(obj):
    if isinstance(obj, np.generic):
        return np.asscalar(obj)
    else:
        return obj
        
def create_doc(args, strategy, sargs_str, train_stats, dev_stats, test_stats, feature_desc=None):
    doc_json = dict()
    params = copy.deepcopy(args)
    del params['strategy']
    if hasattr(getattr(strategies, strategy), 'parser'):
        parser = getattr(strategies, strategy).parser
        sargs = parse(parser, sargs_str.split())
    else:
        sargs = {}
    all_args_str = " ".join(["--" + arg + " " + str(val) for arg, val in sargs.items()])
    sargs['cmd_line'] = all_args_str
    doc_json = {
        'timestamp': datetime.datetime.now(),
        'common_args': params,
        'strategy': strategy,
        'strategy_args': sargs,
        'trains_stats': {k: _numpy_to_pytype(v) for k, v in train_stats.items()},
        'dev_stats': {k: _numpy_to_pytype(v) for k, v in dev_stats.items()},
        'test_stats': {k: _numpy_to_pytype(v) for k, v in test_stats.items()},
        'feature_desc': feature_desc
    }
    return doc_json