from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as npy
import data_loader as data
import models
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dir = os.path.dirname(__file__)


def init():
    dropout = 0.6
    learning_rate = 0.01
    snapshot_step = 500  # steps to snapshot
    epoch = 200000
    batch_size = 64
    run_id = 'speech_cnn_' + str(int(time.time()))

    # Inputs shaping
    width = 20  # mfcc featu
    height = 80  # (max) length of utterance
    classes = 10  # digits
    input_shape = [-1, width, height, 1]

    # Fetch inputs
    batch = data.mfcc_batch_generator(batch_size)
    X, Y = next(batch)
    X = npy.reshape(X, input_shape)
    testX, testY = next(batch)
    testX = npy.reshape(testX, input_shape)

    # Instantiante model for test
    # accX, accY = next(batch)
    # accX = npy.hape(accX, input_shape)
    # model_test = models.create_model(learning_rate, [None, width, height, 1], classes)

    # Instantiante model for training
    model = models.create_model(learning_rate, [None, width, height, 1], classes, dir, drop=dropout)

    # evaluate_model_accuracy(model, testX, testY)
    model.fit({'input': X}, {'target': Y}, n_epoch=epoch, validation_set=({'input': testX}, {'target': testY}),
              show_metric=True, snapshot_epoch=False, snapshot_step=snapshot_step, run_id=run_id)


# Test model accuracry
def evaluate_model_accuracy(model, X, Y):
    good_ = 0
    i = 0
    batch_size = len(X)
    while i < batch_size:
        # Run the model on one example
        prediction = model.predict([X[i]])
        label = npy.argmax(Y[i])
        # print("Prediction: %s" % str(prediction[0]))
        pred = npy.argmax(prediction[0])
        i = i + 1
        if (label == pred):
            good_ = good_ + 1
    print("Accuracy test batch size :" + str(batch_size))
    print("Total good reco :" + str(good_))
    print("Reco avg: %s " % str(round((good_ * 100) / (batch_size))) + "%")


if __name__ == "__main__":
    init()
