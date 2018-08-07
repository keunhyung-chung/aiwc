import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Input, Concatenate, Lambda
from keras import backend as K

import sys
from replay_buffer import ReplayBuffer

# tensoflow
# DDPG Agent for the soccer robot 
# continous action control
class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_max):
        # load model if True
        self.load_model = False

        tf.reset_default_graph()
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))        

        # information of state and action
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_max = float(action_max)
        self.action_min = -float(action_max)

        # hyper parameters
        self.h_critic = 16
        self.h_actor = 16
        self.lr_critic = 1e-3
        self.lr_actor = 1e-4
        self.discount_factor = 0.99        
        self.tau = 0.01 # soft target update rate

        self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim])
        self.reward_ph = tf.placeholder(dtype=tf.float32, shape=[None])
        self.next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim])
        self.done_ph = tf.placeholder(dtype=tf.float32, shape=[None])

        with tf.variable_scope('actor'):
            self.action = self.generate_actor_network(self.state_ph, True)
        with tf.variable_scope('target_actor'):
            self.target_action = self.generate_actor_network(self.next_state_ph, False)
        with tf.variable_scope('critic'):
            self.qvalue = self.generate_critic_network(self.state_ph, self.action, True)
        with tf.variable_scope('target_critic'):
            self.target_qvalue = self.generate_critic_network(self.next_state_ph, self.target_action, False)

        self.a_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')
        self.ta_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_actor')
        self.c_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')
        self.tc_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_critic')

        q_target = tf.expand_dims(self.reward_ph, 1) + self.discount_factor * self.target_qvalue * (1 - tf.expand_dims(self.done_ph, 1))
        td_errors = q_target - self.qvalue
        critic_loss = tf.reduce_mean(tf.square(td_errors))
        self.train_critic = tf.train.AdamOptimizer(self.lr_critic).minimize(critic_loss, var_list = self.c_params)

        actor_loss = - tf.reduce_mean(self.qvalue)
        self.train_actor = tf.train.AdamOptimizer(self.lr_actor).minimize(actor_loss, var_list = self.a_params)

        self.soft_target_update = [[tf.assign(ta, (1-self.tau) * ta + self.tau * a), tf.assign(tc, (1-self.tau) * tc + self.tau * c)]
                                    for a, ta, c, tc in zip(self.a_params, self.ta_params, self.c_params, self.tc_params)]        

        # exploration
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0
        self.exploration_steps = 100000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) / self.exploration_steps
        self.noise = np.zeros(action_dim)

        self.minibatch_size = 32
        self.pre_train_step = 3
        self.replay_buffer = ReplayBuffer(buffer_size=1000000, minibatch_size=self.minibatch_size)

        self.mu = 0
        self.theta = 0.15
        self.sigma = 0.2

        # tensorboard setting
        self.avg_q_max, self.loss_sum = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/simple_ddpg', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        self.save_file = "./save_model/tensorflow_ddpg-1"
        self.load_file = "./save_model/tensorflow_ddpg-1"
        self.saver = tf.train.Saver()
        if self.load_model:
            self.saver.restore(self.sess, self.load_file)

    def choose_action(self, state):
        return self.sess.run(self.action, feed_dict = {self.state_ph: state[None]})[0]        

    def train_network(self, state, action, reward, next_state, done, step):
        self.sess.run(self.train_critic, feed_dict = {self.state_ph: state,
                                                      self.action: action ,
                                                      self.reward_ph: reward,
                                                      self.next_state_ph: next_state,
                                                      self.done_ph: done})
        self.sess.run(self.train_actor, feed_dict = {self.state_ph: state})
        self.sess.run(self.soft_target_update)

    def generate_critic_network(self, state, action, trainable):

        hidden1 = tf.layers.dense(tf.concat([state,action], axis=1), self.h_critic, activation=tf.nn.relu, trainable=trainable)
        hidden2 = tf.layers.dense(hidden1, self.h_critic, activation=tf.nn.relu, trainable=trainable)
        hidden3 = tf.layers.dense(hidden2, self.h_critic, activation=tf.nn.relu, trainable=trainable)

        qvalue = tf.layers.dense(hidden3, 1, trainable=trainable)

        return qvalue

    def generate_actor_network(self, state, trainable):
        hidden1 = tf.layers.dense(state, self.h_actor, activation=tf.nn.relu, trainable=trainable)
        hidden2 = tf.layers.dense(hidden1, self.h_actor, activation=tf.nn.relu, trainable=trainable)
        hidden3 = tf.layers.dense(hidden2, self.h_actor, activation=tf.nn.relu, trainable=trainable)

        non_scaled_action = tf.layers.dense(hidden3, self.action_dim, activation=tf.nn.sigmoid, trainable=trainable)
        action = non_scaled_action * (self.action_max - self.action_min) + self.action_min

        return action

    def get_action(self, obs):
        # 최적의 액션 선택 + Exploration (Epsilon greedy)

        action = self.choose_action(obs)
        self.printConsole("origianl action: " + str(action))

        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        self.printConsole("noise scale: " + str(self.epsilon))
        self.noise = self.ou_noise(self.noise) 
        self.printConsole("             noise: " + str(self.noise * (self.action_max - self.action_min)/2 * max(self.epsilon, 0)))
        action = action + self.noise * (self.action_max - self.action_min)/2 * max(self.epsilon, 0)
        action = np.maximum(action, self.action_min)
        action = np.minimum(action, self.action_max)

        return action

    def train_agent(self, obs, action, reward, obs_next, done, step):

        self.replay_buffer.add_to_memory((obs, action, reward, obs_next, done))

        if len(self.replay_buffer.replay_memory) < self.minibatch_size * self.pre_train_step:
            return None

        minibatch = self.replay_buffer.sample_from_memory()
        s, a, r, ns, d = map(np.array, zip(*minibatch))

        self.train_network(s, a, r, ns, d, step)
        return None

    # make summary operators for tensorboard
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)
        episode_total_score = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)
        tf.summary.scalar('Total Score/Episode', episode_total_score)

        summary_vars = [episode_total_reward, episode_avg_max_q, episode_avg_loss, episode_total_score]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def ou_noise(self, x):
        return x + self.theta * (self.mu-x) + self.sigma * np.random.randn(self.action_dim)

    def printConsole(self, message):
        print(message)
        sys.__stdout__.flush()

