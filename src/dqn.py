import numpy as np
import random
import tensorflow as tf

from collections import deque


class DQN:
    FINAL_EPSILON = 0.1
    INITIAL_EPSILON = 1.0

    def __init__(self, actions, max_replay_memory, observe,
                 explore, statuses, game_name="game"):
        self.actions = actions
        self.max_replay_memory = max_replay_memory
        self.observe = observe
        self.explore = explore
        self.statuses = statuses
        self.game_name = game_name

        self.batch_size = 32
        self.gamma = 0.9
        self.epsilon = self.INITIAL_EPSILON
        self.replay_memory = deque()
        self.image_size = 84
        self.current_state = None
        self.input_layer, self.output_layer = self.create_network
        self.time_step = 0
        self.checkpoint_path = "saved_networks"

        # define cost function
        self.action_input = tf.placeholder(tf.float32, [None, self.actions])
        self.target = tf.placeholder(tf.float32, [None])
        q_action = tf.reduce_sum(tf.multiply(self.output_layer, self.action_input),
                                 reduction_indices=1)
        cost = tf.reduce_mean(tf.square(self.target - q_action))
        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

        # saving and loading nn
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state(self.checkpoint_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    @property
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
        input_layer = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 4])
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
        shape = conv_layer3.get_shape()

        # print("dimension: {0}".format(str(shape[1] * shape[2] * shape[3]))) = 2304
        conv_layer3_2d = tf.reshape(conv_layer3, [-1, 2304])

        # fourth full-connected layer
        fc_layer4 = tf.nn.relu(tf.matmul(conv_layer3_2d, layer4_weights) + layer4_bias)

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
        q_value_batch = self.output_layer.eval(feed_dict={self.input_layer: next_state_batch})
        for i in range(0, self.batch_size):
            terminal = minibatch[i][self.statuses["terminate"]]
            if terminal:
                target_batch.append(reward_batch[i])
            else:
                target_batch.append(reward_batch * self.gamma * np.max(q_value_batch[i]))

        # perform gradient step
        self.train_step.run(feed_dict={
            self.action_input: action_batch,
            self.target: target_batch,
            self.input_layer: state_batch
        })

        # saving network
        if self.time_step % 10000 == 0:
            self.saver.save(self.session, "saved_networks/" + self.game_name + "-dqn",
                            global_step=self.time_step)

    def get_action(self):
        q_value = self.output_layer.eval(feed_dict={self.input_layer: [self.current_state]})[0]
        action = np.zeros(self.actions)
        action_index = 0
        if random.random() <= self.epsilon:
            action_index = random.randrange(self.actions)
            action[action_index] = 1
        else:
            action_index = np.argmax(q_value)
            action[action_index] = 1

        if self.epsilon > self.FINAL_EPSILON and self.time_step > self.observe:
            self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.explore

        return action

    def fitc(self, next_state, action, reward, terminal):
        # the observe of the environment
        new_state = np.append(next_state, self.current_state[:, :, 1:], axis=2)
        self.replay_memory.append((self.current_state, action, reward, new_state, terminal))

        if len(self.replay_memory) > self.max_replay_memory:
            self.replay_memory.popleft()

        # stop exploring and begin training
        if self.time_step > self.observe:
            self.train()

        self.log()

        self.current_state = new_state
        self.time_step += 1

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
        return tf.nn.conv2d(x, weights, strides=[1, stride, stride, 1], padding="SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def log(self):
        state = ""
        if self.time_step <= self.observe:
            state = "observe"
        elif self.observe < self.time_step <= self.observe + self.explore:
            state = "explore"
        else:
            state = "train"

        message = "timestamp: {0}, state: {1}, epsilon: {2}".format(self.time_step,
                                                                    state,
                                                                    self.epsilon)
        print(message)


def main():
    pass

if __name__ == "__main__":
    main()
