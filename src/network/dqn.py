import numpy as np
import os
import random
import sys
import tensorflow as tf

from collections import deque

class DQN:
    def __init__(self, actions, replay_memory, observe, explore, game_name="game"):
        self.actions = actions
        self.replay_memory = replay_memory
        self.observe = observe
        self.explore = explore
        self.game_name = game_name

        self.batch_size = 32
        self.gamma = 0.9
        self.epsilon = 1.0
        self.image_size = 84
        self.session = tf.Session()
        self.current_state = None

    def create_network(self):
        # network weights

        layer1_weights = self.init_weights([8, 8, 4, 32])
        layer1_bias = self.init_biases([32])

        layer2_weigths = self.init_weights([4, 4, 32, 64])
        layer2_bias = self.init_biases([64])

        layer3_weights = self.init_weights([3, 3, 64, 64])
        layer3_bias = self.init_biases([64])

        layer4_weights = self.init_weights([1600, 512])
        layer4_bias = self.init_biases([512])

        layer5_weights = self.init_weights([512, self.actions])
        layer5_bias = self.init_biases([self.actions])

        # model nn
        input_layer = tf.placeholder(tf.float32, [None, self.image_size,
                                                  self.image_size, 4])
        # first convolutional layer: 32 filters, 8x8, 4 strides
        conv_layer1 = tf.nn.relu(self.conv2d(input_layer, layer1_weights, 4) + layer1_bias)
        pooling1 = self.max_pool_2x2(conv_layer1)

        # second convolutional layer: 64 filters, 4x4, 2 strides
        conv_layer2 = tf.nn.relu(self.conv2d(pooling1, layer2_weigths, 2) + layer2_bias)

        # third convolutional layer: 64 filters, 3x3, 1 stride
        conv_layer3 = tf.nn.relu(self.conv2d(conv_layer2, layer3_weights, 1) + layer3_bias)
        # reshape to 2d tensor
        shape = conv_layer3.get_shape()

        ####################################
        # check out shape of new tensor!!! #
        ####################################
        conv_layer3_2d = tf.reshape(conv_layer3, [shape[0], shape[1] * shape[2] * shape[3]])

        # fourth full-connected layer
        fc_layer4 = tf.nn.relu(tf.matmul(conv_layer3_2d, layer4_weights) + layer4_bias)

        # fifth full-connected layer: Q-value
        output_layer = tf.matmul(fc_layer4, layer5_weights) + layer5_bias

        return input_layer, output_layer

    def train(self):
        input_layer, output_layer = self.create_network()

        # define cost function
        action = tf.placeholder(tf.float32, [None, self.actions])
        target = tf.placeholder(tf.placeholder(tf.float32, [None]))
        readout_action = tf.reduce_sum(tf.multiply(output_layer, action), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(target - readout_action))
        train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)



        # logs

        # do nothing state and preprocess the image to 84x84
        actions = np.zeros(self.actions)

    def set_init_state(self, observation):
        self.current_state = np.stack((observation, observation,
                                       observation, observation),
                                      axis=2)


    def run(self, next_observation, action, reward):
        pass


    def init_weights(self, shape):
        weights = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(weights)

    def init_biases(self, shape):
        biases = tf.constant(0.01, shape=shape)
        return tf.Variable(biases)

    def conv2d(self, x, weights, stride):
        return tf.nn.conv2d(x, weights, strides=[1, stride, stride, 1], padding="SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")



def main():
    pass

if __name__ == "__main__":
    main()
