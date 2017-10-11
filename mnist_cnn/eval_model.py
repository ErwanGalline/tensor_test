from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as npy
import tflearn.datasets.mnist as mnist

import models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dir = os.path.dirname(__file__)


def init():
    learning_rate = 0.01
    classes = 10  # digits

    # Fetch inputs
    X, Y,testX, testY = mnist.load_data(one_hot=True)
    testX = testX.reshape([-1, 28, 28, 1])

    # Instantiante model for testing
    model = models.create_model(learning_rate, [None, 28, 28, 1], 10, dir)

    model.load(dir + "/checkpoints/step-17200")
    evaluate_model_accuracy(model, testX, testY)


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
    print("Reco avg: %s " % str(((good_ * 100) / (batch_size))) + "%")


if __name__ == "__main__":
    init()
