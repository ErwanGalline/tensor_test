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
    dropout = 0.8
    learning_rate = 0.005
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

    # Instantiante model for training
    model = models.create_model(learning_rate, [None, width, height, 1], classes, dir, drop=dropout)

    model.load(dir + "/checkpoints/" + "step-20500")
    # evaluate_model_accuracy(model, testX, testY)
    model.fit({'input': X}, {'target': Y}, n_epoch=epoch, validation_set=None,
              show_metric=True, snapshot_epoch=False, snapshot_step=snapshot_step, run_id=run_id)


if __name__ == "__main__":
    init()
