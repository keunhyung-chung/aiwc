import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Input, Concatenate
from keras import backend as K

# DDPG Agent for the soccer robot 
# continous action control
class DDPGAgent:
    def __init__(self, state_size, action_size):
        # load model if True
        self.load_model = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.history_size = 4
        # store the history and the action pair
        self.action = np.zeros(action_size)
        self.history = np.zeros([1, self.state_size, self.history_size])

        # learning weights of actor and critic
        self.lr_actor = 0.0001
        self.lr_critic = 0.001
        self.tau = 0.001

        # parameters about epsilon
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.
        self.exploration_steps = 100000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps

        # parameters about training
        self.batch_size = 32
        self.train_start = 32
        self.discount_factor = 0.99
        self.memory = deque(maxlen=100000)
        #self.no_op_steps = 30

        # build model
        self.critic = self.build_critic_network()
        self.target_critic = self.build_critic_network()
        self.actor = self.build_actor_network()
        self.target_actor = self.build_actor_network()

        self.actor_trainer = self.actor_trainer()

        # tensorboard setting
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.loss_sum = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/goalie_ddpg', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.critic.load_weights("./save_model/goalie_ddpg_critic.h5")
            self.actor.load_weights("./save_model/goalie_ddpg_actor.h5")

        self.target_critic.set_weights(self.critic.get_weights())
        self.target_actor.set_weights(self.actor.get_weights())

    # critic network
    def build_critic_network(self):
        input_state = Input(shape=[self.state_size*self.history_size])
        input_action = Input(shape=[self.action_size])
        s0 = Dense(256, activation='relu')(input_state)
        s1 = Dense(256, activation='linear')(s0)
        a0 = Dense(256, activation='linear')(input_action)
        c0 = Concatenate()([s1, a0])
        c1 = Dense(256, activation='relu')(c0)
        c2 = Dense(1, activation='linear')(c1)

        critic = Model(inputs=[input_state, input_action], outputs=c2)
        critic.compile(loss='mse', optimizer=Adam(lr=self.lr_critic))
        critic.summary()

        return  critic

    # policy network
    def build_actor_network(self):
        input_state = Input(shape=[self.state_size*self.history_size])
        a0 = Dense(256, activation='relu')(input_state)
        a1 = Dense(256, activation='relu')(a0)
        action = Dense(self.action_size,activation='tanh')(a1)

        actor = Model(inputs=input_state, outputs=action)
        actor.summary()

        return actor

    def actor_trainer(self):
        states = Input(shape=[self.state_size*self.history_size])
        actions = self.actor(states)

        critic_output = self.critic([states, actions])

        loss = -K.mean(critic_output)
        optimizer = Adam(lr=self.lr_actor)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([states], [], updates=updates)

        return train

    def update_target_network(self):
        critic_weights = self.critic.get_weights()
        target_critic_weights = self.target_critic.get_weights()
        actor_weights = self.actor.get_weights()
        target_actor_weights = self.target_actor.get_weights()

        for i in range(len(critic_weights)):
            target_critic_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * target_critic_weights[i]
        self.target_critic.set_weights(target_critic_weights)

        for i in range(len(actor_weights)):
            target_actor_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * target_actor_weights[i]
        self.target_actor.set_weights(target_actor_weights)

    def get_action(self, history):
        action = self.actor.predict(history)[0]
        noise = max(self.epsilon, 0) * self.ou_noise(action,  0.0 , 0.60, 0.30)

        return action + noise

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, history, action, reward, next_history, reset):
        self.memory.append((history, action, reward, next_history, reset))

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size, self.history_size))
        next_history = np.zeros((self.batch_size, self.state_size, self.history_size))
        target = np.zeros((self.batch_size,))
        actions, reward, reset = [], [], []

        for i in range(self.batch_size):
            history[i] = mini_batch[i][0]
            next_history[i] = mini_batch[i][3]
            actions.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            reset.append(mini_batch[i][4])

        history = np.reshape(history, [self.batch_size, -1])
        next_history = np.reshape(next_history, [self.batch_size, -1])
        actions = np.reshape(actions, [self.batch_size, -1])        
        target_value = self.target_critic.predict([next_history, self.target_actor.predict(next_history)])

        for i in range(self.batch_size):
            if reset[i]:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.discount_factor * target_value[i]

        self.loss_sum += self.critic.train_on_batch([history, actions], target)
        self.actor_trainer([history])

        self.update_target_network()

    def save_model(self, name):
        self.critic.save_weights(name + '_critic.h5')
        self.actor.save_weights(name + "_actor.h5")

    # make summary operators for tensorboard
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        # episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)
        episode_total_score = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        # tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)
        tf.summary.scalar('Total Score/Episode', episode_total_score)

        summary_vars = [episode_total_reward, episode_avg_max_q, episode_avg_loss, episode_total_score]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def ou_noise(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(self.action_size)

