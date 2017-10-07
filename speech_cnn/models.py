from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


# Create model
# Param : dropout value , learning rate , input shape
def create_model(learning_rate, input_shape, nb_classes, base_path, drop=1):
    network = input_data(shape=input_shape, name='input')
    network = conv_2d(network, 64, [4, 16], activation='relu')
    # network = max_pool_2d(network, 2)
    # network = local_response_normalization(network)
    # network = conv_2d(network, 64, [2, 8], activation='relu')
    # network = max_pool_2d(network, 2)
    # network = local_response_normalization(network)
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, drop)
    network = fully_connected(network, 64, activation='relu')
    network = dropout(network, drop)
    network = fully_connected(network, nb_classes, activation='softmax')
    network = regression(network, optimizer='sgd', learning_rate=learning_rate,
                         loss='categorical_crossentropy', name='target')
    model = tflearn.DNN(network, tensorboard_verbose=3, tensorboard_dir=base_path + "/tflearn_logs/",
                        checkpoint_path=base_path + "/checkpoints/step")
    return model
