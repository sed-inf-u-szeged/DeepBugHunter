import tensorflow as tf

# Matthew's correlation coefficient
def mcc(labels, preds):
    tp, tp_update = tf.metrics.true_positives(labels, preds)
    tn, tn_update = tf.metrics.true_negatives(labels, preds)
    fp, fp_update = tf.metrics.false_positives(labels, preds)
    fn, fn_update = tf.metrics.false_negatives(labels, preds)
    eps = 1e-7

    num = (tp * tn - fp * fn)
    denom = tf.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)) + eps

    return (num / denom), tf.group(tp_update, tn_update, fp_update, fn_update)

# F-Measure
def fmes(labels, preds):
    precision, precision_update = tf.metrics.precision(labels, preds)
    recall, recall_update = tf.metrics.recall(labels, preds)
    eps = 1e-7

    num = (2 * precision * recall)
    denom = precision + recall + eps

    return (num / denom), tf.group(precision_update, recall_update)