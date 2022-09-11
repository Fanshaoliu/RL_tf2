"""
The Dueling DQN based on this paper: https://arxiv.org/abs/1511.06581

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.random.set_seed(1)


class dueling_layer(tf.keras.layers.Layer):
    def __init__(self, units, activation='relu', **kwargs):
        self.units = units
        self.activation = tf.keras.layers.Activation(activation)
        super(dueling_layer, self).__init__(**kwargs)

    # initialize data
    def build(self, input_shape):
        """构建所需要的参数"""
        self.kernel2 = self.add_weight(name='kernel2',
                                      shape=(input_shape[1], 1),
                                      initializer='uniform',
                                      trainable=True)
        self.bias2 = self.add_weight(name='bias2',
                                    shape=(1),
                                    initializer='zeros',
                                    trainable=True)
        self.kernel1 = self.add_weight(name='kernel1',
                                      shape=(input_shape[1], self.units),
                                      initializer='uniform',
                                      trainable=True)
        self.bias1 = self.add_weight(name='bias1',
                                    shape=(self.units),
                                    initializer='zeros',
                                    trainable=True)
        super(dueling_layer, self).build(input_shape)

    def call(self, x):

        return x @ self.kernel1 + self.bias1 + x @ self.kernel2 + self.bias2

class DuelingDQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            dueling=True,
            sess=None,
    ):
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

        self.dueling = dueling      # decide to use dueling DQN or not

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features*2+2))
        if self.dueling:
            self.q_target = tf.keras.models.Sequential([
                # tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(20,input_shape=[self.n_features], activation='relu'),
                # tf.keras.layers.Dropout(0.2),
                dueling_layer(self.n_actions)
            ])
            self.q_eval = tf.keras.models.Sequential([
                # tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(20,input_shape=[self.n_features], activation='relu'),
                # tf.keras.layers.Dropout(0.2),
                dueling_layer(self.n_actions)
            ])
        else:
            self.q_target = tf.keras.models.Sequential([
                # tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(20,input_shape=[self.n_features], activation='relu'),
                # tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(self.n_actions)
            ])
            self.q_eval = tf.keras.models.Sequential([
                # tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(20,input_shape=[self.n_features], activation='relu'),
                # tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(self.n_actions)
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
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:  # choosing action
            actions_value = self.q_eval(observation)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.replace_target_op()
            print('\ntarget_params_replaced\n')

        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
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

        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1





