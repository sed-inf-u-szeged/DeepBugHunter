import numpy as np
import tensorflow as tf

import metrics as mt

# Custom model function for the classifier
def _custom_classifier(features, labels, mode, params):

    # Create L2 regularizer
    reg = tf.keras.regularizers.l2(params['beta'])

    # Create "layers" fully connected layers, each with "neurons" neurons
    net = tf.compat.v1.feature_column.input_layer(features, params['columns'])
    for layer_id in range(params['layers']):
        net = tf.compat.v1.layers.dense(net, units=params['neurons'], activation=tf.nn.relu, kernel_regularizer=reg)

    # Compute logits (1 per class).
    logits = tf.compat.v1.layers.dense(net, params['classes'], activation=None, kernel_regularizer=reg)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Add L2 regularization loss
    loss += tf.compat.v1.losses.get_regularization_loss()

    # Compute evaluation metrics.
    metrics = {
        'accuracy': tf.compat.v1.metrics.accuracy(labels, predicted_classes),
        'precision': tf.compat.v1.metrics.precision(labels, predicted_classes),
        'recall': tf.compat.v1.metrics.recall(labels, predicted_classes),
        'tp': tf.compat.v1.metrics.true_positives(labels, predicted_classes),
        'tn': tf.compat.v1.metrics.true_negatives(labels, predicted_classes),
        'fp': tf.compat.v1.metrics.false_positives(labels, predicted_classes),
        'fn': tf.compat.v1.metrics.false_negatives(labels, predicted_classes),
        'mcc': mt.mcc(labels, predicted_classes),
        'fmes': mt.fmes(labels, predicted_classes)
    }

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def create_classifier(args):
    # Build a custom classifier
    return tf.estimator.Estimator(
        model_fn=_custom_classifier,
        params=args,
        config=tf.estimator.RunConfig(
            # Repeatability!
            tf_random_seed=args['seed'],
            # Save checkpoints exactly once per epoch
            save_checkpoints_steps=args['steps_per_epoch'],
            device_fn=lambda op: args['device'],
            session_config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=args['log_device'])
        ),
        model_dir=args['model_dir'],
        warm_start_from=args['warm_start_dir']
    )
