import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def load_data(file_name, label_name, seed):
    """Returns the whole dataset"""

    # read csv into a dataframe
    data = pd.read_csv(file_name, header=0)

    # deterministic shuffle
    data = shuffle(data, random_state=seed)

    # separate the dataset into features and labels
    features, labels = data, data.pop(label_name)

    return features, labels
