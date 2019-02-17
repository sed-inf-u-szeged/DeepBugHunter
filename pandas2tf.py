import tensorflow as tf

def train_input_fn(data, batch_size):
    """An input function for training"""

    features, labels = data

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Repeat, and batch the examples.
    dataset = dataset.repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(data, batch_size):
    """An input function for evaluation or prediction"""

    inputs = None

    try:
        # If there are labels, then we evaluate
        features, labels = data
        inputs = (dict(features), labels)
    except:
        # If there are no labels --> prediction
        inputs = dict(data)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset
