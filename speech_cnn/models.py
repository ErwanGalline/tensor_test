from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tflearn as tfl

# Create model
# Param : dropout value , learning rate
def create_model(dropout,learning_rate ) :
    network = tfl.input_data(shape=[None, 28, 28, 1], name='input')
    network = tfl.conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = tfl.max_pool_2d(network, 2)
    network = tfl.local_response_normalization(network)
    network = tfl.conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = tfl.max_pool_2d(network, 2)
    network = tfl.local_response_normalization(network)
    network = tfl.fully_connected(network, 128, activation='tanh')
    network = tfl.dropout(network, dropout)
    network = tfl.fully_connected(network, 256, activation='tanh')
    network = tfl.dropout(network,dropout)
    network = tfl.fully_connected(network, 10, activation='softmax')
    network = tfl.regression(network, optimizer='adam', learning_rate=learning_rate,
                         loss='categorical_crossentropy', name='target')

    return network