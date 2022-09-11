"""
The double DQN based on this paper: https://arxiv.org/abs/1509.06461

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.random.set_seed(1)


class DoubleDQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=3000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            double_q=True,
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

        self.double_q = double_q    # decide to use double q or not

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features*2+2))
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
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.q_eval(observation)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        actions_value = self.q_eval(observation)
        action = np.argmax(actions_value)

        if not hasattr(self, 'q'):  # record action value it gets
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q*0.99 + 0.01 * np.max(actions_value)
        self.q.append(self.running_q)

        if np.random.uniform() > self.epsilon:  # choosing action
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


        # print(self.q_eval.trainable_variables)
        with tf.GradientTape() as tape:
            q_next, q_eval4next = self.q_target(batch_memory[:, -self.n_features:], training=True), self.q_eval(
                batch_memory[:, -self.n_features:], training=True)
            q_target = q_eval4next.numpy()
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            eval_act_index = batch_memory[:, self.n_features].astype(int)
            reward = batch_memory[:, self.n_features + 1]

            if self.double_q:
                max_act4next = np.argmax(q_eval4next,
                                         axis=1)  # the action that brings the highest value is evaluated by q_eval
                # print(max_act4next)
                # print(q_next)
                selected_q_next = q_next.numpy()[
                    batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
            else:
                selected_q_next = np.max(q_next, axis=1)  # the natural DQN

            q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next
            loss = self.target_loss(q_target, q_eval4next)
            # print(loss)
        gradients = tape.gradient(loss, self.q_eval.trainable_variables)
        # print(gradients)
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



