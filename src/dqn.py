import numpy as np
import os
import random
import tensorflow as tf

from collections import deque


class DQN:
    SAVE_EVERY_X_STEPS = 10000

    def __init__(self, actions, max_replay_memory, observe,
                 explore, statuses, learning_rate, batch_size,
                 gamma, initial_epsilon, final_epsilon, game_name="game"):
        self.actions = actions
        self.max_replay_memory = max_replay_memory
        self.observe = observe
        self.explore = explore
        self.statuses = statuses
        self.learning_rate = learning_rate
        self.epsilon = self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.game_name = game_name
        self.batch_size = batch_size
        self.gamma = gamma

        self.replay_memory = deque()
        self.image_size = 84
        self.current_state = None
        self.input_layer, self.output_layer = self.create_network()
        self.time_step = 0
        self.checkpoint_path = "saved_networks"

        # form tensorflow graph
        # define cost function
        self.action_input = tf.placeholder(tf.float32, [None, self.actions])
        self.target = tf.placeholder(tf.float32, [None])
        q_value = tf.reduce_sum(tf.multiply(self.output_layer,
                                            self.action_input),
                                reduction_indices=1)
        cost = tf.reduce_mean(tf.square(self.target - q_value))
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

        # saving and loading nn
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state(self.checkpoint_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def create_network(self):
        # network weights
        layer1_weights = self.init_weights([8, 8, 4, 32])
        layer1_bias = self.init_biases([32])

        layer2_weigths = self.init_weights([4, 4, 32, 64])
        layer2_bias = self.init_biases([64])

        layer3_weights = self.init_weights([3, 3, 64, 64])
        layer3_bias = self.init_biases([64])

        layer4_weights = self.init_weights([2304, 512])
        layer4_bias = self.init_biases([512])

        layer5_weights = self.init_weights([512, self.actions])
        layer5_bias = self.init_biases([self.actions])

        # model nn
        input_layer = tf.placeholder(tf.float32, [None, self.image_size,
                                                  self.image_size, 4])
        # first convolutional layer: 32 filters, 8x8, 4 strides
        conv_layer1 = tf.nn.relu(self.conv2d(input_layer, layer1_weights, 4)
                                 + layer1_bias)
        pooling1 = self.max_pool_2x2(conv_layer1)

        # second convolutional layer: 64 filters, 4x4, 2 strides
        conv_layer2 = tf.nn.relu(self.conv2d(pooling1, layer2_weigths, 2)
                                 + layer2_bias)

        # third convolutional layer: 64 filters, 3x3, 1 stride
        conv_layer3 = tf.nn.relu(self.conv2d(conv_layer2, layer3_weights, 1)
                                 + layer3_bias)
        # reshape to 2d tensor
        # print("dimension: {0}".format(str(shape[1] * shape[2] * shape[3])))
        # >>> 2304
        conv_layer3_2d = tf.reshape(conv_layer3, [-1, 2304])

        # fourth full-connected layer
        fc_layer4 = tf.nn.relu(tf.matmul(conv_layer3_2d, layer4_weights)
                               + layer4_bias)

        # fifth full-connected layer: Q-value
        output_layer = tf.matmul(fc_layer4, layer5_weights) + layer5_bias

        return input_layer, output_layer

    def train(self):
        # get random minibatch from replay memory
        minibatch = random.sample(self.replay_memory, self.batch_size)
        state_batch = [data[self.statuses["state"]] for data in minibatch]
        action_batch = [data[self.statuses["action"]] for data in minibatch]
        reward_batch = [data[self.statuses["reward"]] for data in minibatch]
        next_state_batch = [data[self.statuses["next_state"]] for data in minibatch]

        # calculate target value
        target_batch = []
        q_value_batch = self.output_layer.eval(feed_dict={
                                                   self.input_layer:
                                                       next_state_batch
                                               })
        for i in range(0, self.batch_size):
            terminal = minibatch[i][self.statuses["terminate"]]
            if terminal:
                target_batch.append(reward_batch[i])
            else:
                target_batch.append(reward_batch[i] + self.gamma
                                    * np.max(q_value_batch[i]))

        # perform gradient step
        self.optimizer.run(feed_dict={
            self.action_input: action_batch,
            self.target: target_batch,
            self.input_layer: state_batch
        })

    def get_action(self):
        q_value = self.output_layer.eval(feed_dict={
                                            self.input_layer: [
                                                self.current_state
                                                ]
                                         })[0]
        action = np.zeros(self.actions)
        action_index = 0
        if random.random() <= self.epsilon:
            action_index = random.randrange(self.actions)
        else:
            action_index = np.argmax(q_value)
        action[action_index] = 1

        if self.epsilon > self.final_epsilon and self.time_step > self.observe:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) \
                            / self.explore

        return action, q_value

    def fit(self, next_state, action, reward, terminal):
        # the observe of the environment
        new_state = np.append(next_state, self.current_state[:, :, :3], axis=2)

        # store the state im memory
        self.replay_memory.append((self.current_state, action, reward,
                                   new_state, terminal))
        if len(self.replay_memory) > self.max_replay_memory:
            self.replay_memory.popleft()

        # stop exploring and begin training
        if self.time_step > self.observe:
            self.train()

        self.current_state = new_state
        self.time_step += 1

        # saving network
        if self.time_step % self.SAVE_EVERY_X_STEPS == 0:
            self.saver.save(self.session, self.checkpoint_path + "/" \
                                          + self.game_name + "-dqn",
                            global_step=self.time_step)

    def set_init_state(self, observation):
        self.current_state = np.stack((observation, observation,
                                       observation, observation),
                                      axis=2)

    def init_weights(self, shape):
        weights = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(weights)

    def init_biases(self, shape):
        biases = tf.constant(0.01, shape=shape)
        return tf.Variable(biases)

    def conv2d(self, x, weights, stride):
        return tf.nn.conv2d(x, weights, strides=[1, stride, stride, 1],
                            padding="SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding="SAME")


def main():
    pass

if __name__ == "__main__":
    main()
