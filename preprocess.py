import math
import logging
import pandas as pd
from sklearn.utils import shuffle
from sklearn.utils import resample as resamp
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger('DeepBugHunter')

def normalize(features, labels):
    # map to the [0,1] interval
    return features[features.columns].apply(lambda col: (col - col.min()) / (col.max() - col.min())), labels

def standardize(features, labels):
    # map to a Gaussian distribution centered at 0
    return features[features.columns].apply(lambda col: (col - col.mean()) / col.std()), labels

def binarize(features, labels):
    # store original labels
    global orig_labels
    orig_labels = {features.index[i]:labels.iloc[i] for i in range(len(features))}
    
    # map to boolean
    return features, labels.map(lambda x: 1 if x > 0 else 0)

def get_orig_labels(features):
    return [orig_labels[features.index[i]] for i in range(len(features))]

def resample(features, labels, mode, amount, seed):

    mode = mode == 'up'

    # reconstruct data
    name = labels.name
    data = features
    data.insert(len(data.columns), name, labels)

    # split to classes
    bins = []
    cnt = 0 if mode else math.inf
    for x in data[name].unique():
        _bin = data[data[name] == x]
        _len = len(_bin)
        cnt = (max if mode else min)(cnt, _len)
        bins.append(_bin)

    # resample relevant classes
    for i, item in enumerate(bins):
        current = len(item)
        diff = int(abs(current - cnt) * (amount / 100))
        if diff:
            new_cnt = current + diff if mode else current - diff
            bins[i] = resamp(bins[i], replace=True, n_samples=new_cnt, random_state=seed)

    # recombine classes to the whole dataset (and reshuffle!)
    data = pd.concat(bins, ignore_index=True)
    data = shuffle(data, random_state=seed)

    # return feature/label split
    return data, data.pop(name)
        

def split(data, seed, folds=10):

    features, labels = data

    # init a 10-fold cross validation
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    # return a generator for the train/test splits per fold
    def fold_generator():
        for train_indices, test_indices in kf.split(features, labels):
            yield (features.iloc[train_indices], labels.iloc[train_indices]), (features.iloc[test_indices], labels.iloc[test_indices])

    return fold_generator