from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as npy
import data_loader as data
import models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dir = os.path.dirname(__file__)


def init():
    learning_rate = 0.01
    batch_size = 5
    # Inputs shaping
    width = 20  # mfcc featu
    height = 80  # (max) length of utterance
    classes = 10  # digits
    input_shape = [-1, width, height, 1]

    # Fetch inputs
    batch = data.mfcc_batch_generator(batch_size, dir_path="data/spoken_numbers_pcm/")
    testX, testY = next(batch)
    testX = npy.reshape(testX, input_shape)

    # Instantiante model for testing
    model = models.create_model(learning_rate, [None, width, height, 1], classes, dir, model_type="1conv")
    model.load(dir + "/checkpoints_model_1/step-1000")
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
            print("Predicted :" + str(pred))
            print("Real :" + str(label))
            good_ = good_ + 1
    print("Accuracy test batch size :" + str(batch_size))
    print("Total good reco :" + str(good_))
    print("Reco avg: %s " % str(round((good_ * 100) / (batch_size))) + "%")


if __name__ == "__main__":
    init()
