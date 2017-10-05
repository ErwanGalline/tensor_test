from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import numpy as npy
import tflearn as tfl
import data_loader as data
import models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(_):
    dropout = 0.5
    learning_rate = 0.0005
    snapshot_step = 500  # steps to snapshot
    epoch = 500
    batch_size = 100
    runId = 'speech_cnn'

    width = 20  # mfcc features
    height = 80  # (max) length of utterance
    classes = 10  # digits

    batch = word_batch = data.mfcc_batch_generator(batch_size)
    X, Y = next(batch)
    X = tfl.reshape(X, [-1, 20, 80, 1])
    testX, testY = next(batch)
    testX = tfl.reshape(testX, [-1, 20, 80, 1])

    # Instantiante model
    network = models.create_model(dropout, learning_rate)

    model = tfl.DNN(network, tensorboard_verbose=0)
    evaluate_model_accuracy(model , next(batch) )
    model.fit({'input': X}, {'target': Y}, n_epoch=epoch, validation_set=({'input': testX}, {'target': testY}),
              show_metric=True, batch_size=batch_size, shuffle=True,
              snapshot_epoch=False, snapshot_step=snapshot_step,
              validation_batch_size=batch_size, run_id=runId)


# Test model accuracry
def evaluate_model_accuracy(model, X, Y):
    good_res = 0
    batch_size = X.len
    i = 0
    while i < batch_size:
        # Run the model on one example
        prediction = model.predict([X[i]])
        label = npy.argmax(Y[i])
        print("Real label : " + str(label))
        # print("Prediction: %s" % str(prediction[0]))
        predRes = npy.argmax(prediction[0])
        print("Prediction: %s" % str(predRes))
        i = i + 1
        if (label == predRes):
            print("Reco OK !")
            good_res = good_res + 1
        print("=============")
    print("---------------")
    print("Reco avg: %s" % str(good_res / (batch_size)))
