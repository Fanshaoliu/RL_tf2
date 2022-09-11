"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import copy


np.random.seed(1)
tf.random.set_seed(1)

# os.environ['CUDA_VISIBLE_DEVICES'] = "0" #1卡

# gpus = tf.config.experimental.list_physical_devices(device_type='gpu')
# print(gpus)
# # config = tf.compat.v1.ConfigProto
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# # config.gpu_options.per_process_gpu_memory_fraction = 0.2

physical_devices = tf.config.experimental.list_physical_devices('GPU')

if len(physical_devices) > 0:

    for k in range(len(physical_devices)):

        tf.config.experimental.set_memory_growth(physical_devices[k], True)

        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))

    else:

        print("Not enough GPU hardware devices available")

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

class build_net(Model):
    def __init__(self, layer_num=1, hidden_size=128, input_size=None, ouput_size=1, action_function='relu', name=None):
        super(build_net, self).__init__()
        self.input_layer = tf.keras.layers.Dense(hidden_size, activation=action_function, input_shape=[input_size])
        self.hidden_layers = [tf.keras.layers.Dense(hidden_size, activation=action_function, input_shape=[hidden_size]) for _ in range(layer_num-1)]
        self.ouput_layer = tf.keras.layers.Dense(ouput_size)

    def call(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.ouput_layer(x)
        return output

class DeepQNetwork(Model):
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
                 ):
        super(DeepQNetwork, self).__init__()
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self.q_target = tf.keras.models.Sequential([
            # tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(20,input_shape=[self.n_features], activation='relu'),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(4)
        ])
        self.q_eval = tf.keras.models.Sequential([
            # tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(20,input_shape=[self.n_features], activation='relu'),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(4)
        ])

        self.target_loss = tf.keras.losses.MeanSquaredError()
        self.eval_optimizer = tf.keras.optimizers.RMSprop(0.001)

        self.cost_his = []


    def replace_target_op(self):
        t_params = self.q_target.trainable_variables
        e_params = self.q_eval.trainable_variables
        for t, e in zip(t_params, e_params):
            tf.compat.v1.assign(t, e)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.q_eval(observation)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.replace_target_op()

            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        with tf.GradientTape() as tape:
            q_next, q_eval = self.q_target(batch_memory[:, -self.n_features:]), self.q_eval(
                batch_memory[:, :self.n_features], training=True)
            q_target = q_eval.numpy()
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            eval_act_index = batch_memory[:, self.n_features].astype(int)
            reward = batch_memory[:, self.n_features + 1]

            q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
            loss = self.target_loss(q_target, q_eval)
        gradients = tape.gradient(loss, self.q_eval.trainable_variables)
        self.eval_optimizer.apply_gradients(zip(gradients,  self.q_eval.trainable_variables))
        self.cost = loss.numpy()

        # train eval network
        # _, self.cost = self.sess.run([self._train_op, self.loss],
        #                              feed_dict={self.s: batch_memory[:, :self.n_features],
        #                                         self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
