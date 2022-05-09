from keras.models import Sequential
from keras.layers import Dense, Lambda
from keras.optimizers import adam_v2
from keras import backend as K
import tensorflow as tf
import argparse
import dbh_util as util

CLASSES = 2


parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=int, help='Number of layers')
parser.add_argument('--neurons', type=int, help='Number of neurons per layer')
parser.add_argument('--batch', type=int, help='Batch size')
parser.add_argument('--epochs', type=int, help='Epoch count')
parser.add_argument('--lr', type=float, help='Starting learning rate')
parser.add_argument('--beta', type=float, default=0.0, help='L2 regularization bias')


def custom_f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = TP / (Positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = TP / (Pred_Positives + K.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def predict(classifier, test, args, sargs_str, threshold=None):
    sargs = util.parse(parser, sargs_str.split())
    preds = classifier.predict(test[0])
    if threshold is not None:
        preds = [1 if x >= threshold else 0 for x in preds]
    return preds


def create_model(sargs, input_dim):
    reg = tf.keras.regularizers.l2(sargs["beta"])
    model = Sequential()
    model.add(Dense(sargs["neurons"], input_dim=input_dim, activation='relu', kernel_regularizer=reg))

    for _ in range(1, sargs['layers']):
        model.add(Dense(1024, activation='relu', kernel_regularizer=reg))

    model.add(Dense(CLASSES, activation='softmax', kernel_regularizer=reg))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam_v2.Adam(learning_rate=sargs["lr"]), metrics=['acc', custom_f1])
    return model


def learn(train, dev, test, args, sargs_str):
    sargs = util.parse(parser, sargs_str.split())

    model = create_model(sargs, len(train[0].keys()))
    model.fit(train[0], train[1], epochs=sargs["epochs"], batch_size=sargs["batch"])
    model.add(Lambda(lambda x: K.cast(K.argmax(x), dtype='float32'), name='y_pred'))

    train_res = util.sklearn_eval(model, train)
    dev_res = util.sklearn_eval(model, dev)
    test_res = util.sklearn_eval(model, test)
    return train_res, dev_res, test_res, model
