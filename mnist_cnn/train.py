from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tflearn.datasets.mnist as mnist
import models
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dir = os.path.dirname(__file__)


def init():
    dropout = 0.8
    learning_rate = 0.01
    run_id = 'mnist_cnn_' + str(int(time.time()))

    X, Y, testX, testY = mnist.load_data(one_hot=True)
    X = X.reshape([-1, 28, 28, 1])
    testX = testX.reshape([-1, 28, 28, 1])

    # Instantiante model for training
    model = models.create_model(learning_rate, [None, 28, 28, 1], 10, dir, drop=dropout)

    # evaluate_model_accuracy(model, testX, testY)
    model.fit({'input': X}, {'target': Y}, n_epoch=20,
              validation_set=({'input': testX}, {'target': testY}), show_metric=True, run_id=run_id)
    model.save(dir + "/saveMnist_trained")


if __name__ == "__main__":
    init()
