#!/usr/bin/python3

# keunhyung 8/12
# shoot and chase
# multi experience

from __future__ import print_function

from twisted.internet import reactor
from twisted.internet.defer import inlineCallbacks

from autobahn.wamp.serializer import MsgPackSerializer
from autobahn.wamp.types import ComponentConfig
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner

import argparse
import random
import math
import os
import sys

import base64
import numpy as np

import helper

import tensorflow as tf

from args import Argument
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
from gym import spaces

#reset_reason
NONE = 0
GAME_START = 1
SCORE_MYTEAM = 2
SCORE_OPPONENT = 3
GAME_END = 4
DEADLOCK = 5

#coordinates
MY_TEAM = 0
OP_TEAM = 1
BALL = 2
X = 0
Y = 1
TH = 2
ACTIVE = 3
TOUCH = 4

class Frame(object):
    def __init__(self):
        self.time = None
        self.score = None
        self.reset_reason = None
        self.coordinates = None

class Component(ApplicationSession):

    def __init__(self, config):
        ApplicationSession.__init__(self, config)

    def printConsole(self, message):
        print(message)
        sys.__stdout__.flush()

    def onConnect(self):
        self.join(self.config.realm)

    @inlineCallbacks
    def onJoin(self, details):

##############################################################################
        def init_variables(self, info):
            # Here you have the information of the game (virtual init() in random_walk.cpp)
            # List: game_time, goal, number_of_robots, penalty_area, codewords,
            #       robot_height, robot_radius, max_linear_velocity, field, team_info,
            #       {rating, name}, axle_length, resolution, ball_radius
            # self.game_time = info['game_time']
            self.field = info['field']
            self.robot_size = 2*info['robot_radius']
            self.goal = info['goal']
            self.max_linear_velocity = info['max_linear_velocity']
            self.number_of_robots = info['number_of_robots']
            self.end_of_frame = False
            self.cur_my_posture = []
            self.cur_op_posture = []
            self.cur_ball = []
            self.pre_ball = [0, 0]

            self.state_dim = 5 # ball, goal, theta
            self.history_size = 2 # frame history size
            self.action_dim = 2 # 2                    
            
            self.arglist = Argument()
            self.obs_shape_n = [(self.state_dim * self.history_size,) for _ in range(1)] # state dimenstion
            self.action_space = [spaces.Discrete(self.action_dim * 2 + 1) for _ in range(1)]
            self.trainers = self.get_trainers(1, self.obs_shape_n, self.action_space, self.arglist)

            # for tensorboard
            self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
            self.summary_writer = tf.summary.FileWriter('summary/aiwc_maddpg', U.get_session().graph)

            U.initialize()
            
            # Load previous results, if necessary
            if self.arglist.load_dir == "":
                self.arglist.load_dir = self.arglist.save_dir
            if self.arglist.display or self.arglist.restore or self.arglist.benchmark:
                print('Loading previous state...')
                U.load_state(self.arglist.load_dir)

            self.episode_rewards = [0.0]  # sum of rewards for all agents
            self.agent_rewards = [[0.0] for _ in range(self.number_of_robots)]  # individual agent reward
            self.final_ep_rewards = []  # sum of rewards for training curve
            self.final_ep_ag_rewards = []  # agent rewards for training curve
            self.agent_info = [[[]]]  # placeholder for benchmarking info
            self.saver = tf.train.Saver()
            self.obs_n = [np.zeros([self.state_dim * self.history_size]) for _ in range(self.number_of_robots)] # histories
            self.train_step = 0
            self.wheels = np.zeros(self.number_of_robots*2)
            self.action_n = [np.zeros(self.action_dim * 2 + 1) for _ in range(self.number_of_robots)] # not np.zeros(2)
                   
            self.save_every_steps = 12000 # save the model every 10 minutes
            self.stats_steps = 6000 # for tensorboard
            self.reward_sum = np.zeros(self.number_of_robots)
            self.score_sum = 0 
            self.active_flag = [[False for _ in range(5)], [False for _ in range(5)]]   
            self.inner_step = 0
            return
##############################################################################
            
        try:
            info = yield self.call(u'aiwc.get_info', args.key)
        except Exception as e:
            self.printConsole("Error: {}".format(e))
        else:
            try:
                self.sub = yield self.subscribe(self.on_event, args.key)
            except Exception as e2:
                self.printConsole("Error: {}".format(e2))
               
        init_variables(self, info)
        
        try:
            yield self.call(u'aiwc.ready', args.key)
        except Exception as e:
            self.printConsole("Error: {}".format(e))
        else:
            self.printConsole("I am ready for the game!")

    def mlp_model(self, input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
        # This model takes as input an observation and returns values of all actions
        with tf.variable_scope(scope, reuse=reuse):
            out = input
            out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
            return out

    def get_trainers(self, num_agent, obs_shape_n, action_space, arglist):
        trainers = []
        model = self.mlp_model
        trainer = MADDPGAgentTrainer

        for i in range(num_agent):
            trainers.append(trainer(
                "agent_%d" % i, model, obs_shape_n, action_space, i, arglist,
                local_q_func=(arglist.good_policy=='ddpg')))
        return trainers

    # make summary operators for tensorboard
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_0_reward = tf.Variable(0.)
        episode_1_reward = tf.Variable(0.)
        episode_2_reward = tf.Variable(0.)
        episode_3_reward = tf.Variable(0.)
        episode_4_reward = tf.Variable(0.)
        episode_total_score = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Reward 0/Episode', episode_0_reward)
        tf.summary.scalar('Reward 1/Episode', episode_1_reward)
        tf.summary.scalar('Reward 2/Episode', episode_2_reward)
        tf.summary.scalar('Reward 3/Episode', episode_3_reward)
        tf.summary.scalar('Reward 4/Episode', episode_4_reward)
        tf.summary.scalar('Total Score/Episode', episode_total_score)

        summary_vars = [episode_total_reward, episode_0_reward, episode_1_reward, 
            episode_2_reward, episode_3_reward, episode_4_reward, episode_total_score]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op
##############################################################################    
    def get_coord(self, received_frame):
        self.cur_ball = received_frame.coordinates[BALL]
        self.cur_my_posture = received_frame.coordinates[MY_TEAM]
        self.cur_op_posture =received_frame.coordinates[OP_TEAM]

    def get_reward(self, reset_reason, i):
        pre_potential = helper.dipole_potential(self.pre_ball[X], self.pre_ball[Y], 2.1, 7)
        cur_potential = helper.dipole_potential(self.cur_ball[X], self.cur_ball[Y], 2.1, 7)
        reward = cur_potential - pre_potential

        # Reset
        if(reset_reason != NONE):
            reward = 0

        reward -= 0.1*helper.distance(self.cur_ball[X], self.cur_my_posture[i][X], 
            self.cur_ball[Y], self.cur_my_posture[i][Y])            
        
        if self.cur_my_posture[i][TOUCH]:
            reward += 20

        if(reset_reason == SCORE_MYTEAM):
            self.score_sum += 1
            reward += 200 # minimum 25
            self.printConsole("my team goal")

        if(reset_reason == SCORE_OPPONENT):
            self.score_sum -= 1
            reward -= 200 # maxmimum -25
            self.printConsole("op team goal")   

        self.printConsole('reward' + str(i) + ': ' + str(reward))
        return reward      

    def pre_processing(self, i):
        relative_ball = helper.rot_transform(self.cur_my_posture[i][X], 
            self.cur_my_posture[i][Y], -self.cur_my_posture[i][TH], 
            self.cur_ball[X], self.cur_ball[Y])
        # self.printConsole('relative ball: ' + str(relative_ball))

        relative_goal = helper.rot_transform(self.cur_my_posture[i][X], 
            self.cur_my_posture[i][Y], -self.cur_my_posture[i][TH], 
            2.05, 0)

        theta = self.cur_my_posture[i][TH]/(math.pi)
        if 1 < theta <= 1.5:
            theta -= 2
        # self.printConsole('theta' + str(i) + ': ' + str(theta))

        return np.concatenate([relative_ball] + [relative_goal] + [[theta]])

    @inlineCallbacks
    def on_event(self, f):        

        @inlineCallbacks
        def set_wheel(self, robot_wheels):
            yield self.call(u'aiwc.set_speed', args.key, robot_wheels)
            return
            
        # initiate empty frame
        received_frame = Frame()

        if 'time' in f:
            received_frame.time = f['time']
        if 'score' in f:
            received_frame.score = f['score']
        if 'reset_reason' in f:
            received_frame.reset_reason = f['reset_reason']
        if 'coordinates' in f:
            received_frame.coordinates = f['coordinates']            
        if 'EOF' in f:
            self.end_of_frame = f['EOF']
        
        #self.printConsole(received_frame.time)
        #self.printConsole(received_frame.score)
        #self.printConsole(received_frame.reset_reason)
        #self.printConsole(self.end_of_frame)
##############################################################################
        if (self.end_of_frame):
            
            # How to get the robot and ball coordinates: (ROBOT_ID can be 0,1,2,3,4)
            #self.printConsole(received_frame.coordinates[MY_TEAM][ROBOT_ID][X])            
            #self.printConsole(received_frame.coordinates[MY_TEAM][ROBOT_ID][Y])
            #self.printConsole(received_frame.coordinates[MY_TEAM][ROBOT_ID][TH])
            #self.printConsole(received_frame.coordinates[MY_TEAM][ROBOT_ID][ACTIVE])
            #self.printConsole(received_frame.coordinates[MY_TEAM][ROBOT_ID][TOUCH])
            #self.printConsole(received_frame.coordinates[OP_TEAM][ROBOT_ID][X])
            #self.printConsole(received_frame.coordinates[OP_TEAM][ROBOT_ID][Y])
            #self.printConsole(received_frame.coordinates[OP_TEAM][ROBOT_ID][TH])
            #self.printConsole(received_frame.coordinates[OP_TEAM][ROBOT_ID][ACTIVE])
            #self.printConsole(received_frame.coordinates[OP_TEAM][ROBOT_ID][TOUCH])
            #self.printConsole(received_frame.coordinates[BALL][X])
            #self.printConsole(received_frame.coordinates[BALL][Y])
                                               
            self.get_coord(received_frame)
##############################################################################
            # Next state, Reward, Reset
            # new_obs_n = [np.zeros([self.state_dim * self.history_size]) for _ in range(self.number_of_robots)]
            new_obs_n = []
            rew_n = []
            done_n = []
            for i in range(self.number_of_robots):
                next_state = self.pre_processing(i)
                # self.printConsole(next_state)
                # new_obs_n[i] = np.append(next_state, next_state - self.obs_n[i][:-self.state_dim]) # position and velocity
                new_obs_n.append(np.append(next_state, next_state - self.obs_n[i][:-self.state_dim])) # position and velocity
                # self.printConsole('observation ' + str(i) + ': '+ str(new_obs_n[i]))

                rew_n.append(self.get_reward(received_frame.reset_reason, i))

                if(received_frame.reset_reason != NONE):
                    done_n.append(True)
                else:
                    done_n.append(False)
            done = all(done_n)
            if done:
                self.printConsole("reset reason: " + str(received_frame.reset_reason))

            # self.printConsole('reward: ' + str(rew_n[0]))
            # rew_n = [sum(rew_n) for i in range(self.number_of_robots)]

            # for i, agent in enumerate(self.trainers):
            #     agent.experience(self.obs_n[i], self.action_n[i], rew_n[i], new_obs_n[i], done_n[i], False)
            for i in range(self.number_of_robots):
                self.trainers[0].experience(self.obs_n[i], self.action_n[i], rew_n[i], new_obs_n[i], done_n[i], False)

            self.obs_n = new_obs_n

            # for i, rew in enumerate(rew_n):
            #     self.episode_rewards[-1] += rew
            #     self.agent_rewards[i][-1] += rew

            # if done:
            #     self.episode_rewards.append(0)
            #     for a in self.agent_rewards:
            #         a.append(0)
            #     self.agent_info.append([[]])
            self.reward_sum += rew_n

            # increment global step counter
            self.train_step += 1

            # update all trainers
            loss = None
            for agent in self.trainers:
                agent.preupdate()
            for agent in self.trainers:
                loss = agent.update(self.trainers, self.train_step)

            # get action
            # self.action_n = [agent.action(obs) for agent, obs in zip(self.trainers,self.obs_n)]
            self.action_n = [self.trainers[0].action(obs) for obs in self.obs_n]
            # self.printConsole("original action: " + str(self.action_n[0]))

            for i in range(self.number_of_robots):
                self.wheels[2*i] = self.max_linear_velocity * (self.action_n[i][1]-self.action_n[i][2]+self.action_n[i][3]-self.action_n[i][4])
                self.wheels[2*i + 1] = self.max_linear_velocity * (self.action_n[i][1]-self.action_n[i][2]-self.action_n[i][3]+self.action_n[i][4])

            # self.printConsole("                 action: " + str(self.wheels[:2]))
            self.printConsole('step: ' + str(self.train_step))

            self.pre_ball = self.cur_ball
            set_wheel(self, self.wheels.tolist())
##############################################################################
            if (self.train_step % self.save_every_steps) == 0:
                U.save_state(self.arglist.save_dir, saver=self.saver)

            # if done: # plot the statics
            if (self.train_step % self.stats_steps) == 0: # plot every 6000 steps (about 5 minuites)
                self.printConsole("add data to tensorboard")
                stats = [sum(self.reward_sum)] + [self.reward_sum[i] for i in range(len(self.reward_sum))] + [self.score_sum]
                for i in range(len(stats)):
                    U.get_session().run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = U.get_session().run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.inner_step)

                self.reward_sum = np.zeros(len(self.reward_sum))
                self.score_sum = 0
                self.inner_step += 1            
##############################################################################
            if(received_frame.reset_reason == GAME_END):
                #(virtual finish() in random_walk.cpp)
                #save your data
                with open(args.datapath + '/result.txt', 'w') as output:
                    #output.write('yourvariables')
                    output.close()
                #unsubscribe; reset or leave  
                yield self.sub.unsubscribe()
                try:
                    yield self.leave()
                except Exception as e:
                    self.printConsole("Error: {}".format(e))

            self.end_of_frame = False
##############################################################################

    def onDisconnect(self):
        if reactor.running:
            reactor.stop()

if __name__ == '__main__':
    
    try:
        unicode
    except NameError:
        # Define 'unicode' for Python 3
        def unicode(s, *_):
            return s

    def to_unicode(s):
        return unicode(s, "utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("server_ip", type=to_unicode)
    parser.add_argument("port", type=to_unicode)
    parser.add_argument("realm", type=to_unicode)
    parser.add_argument("key", type=to_unicode)
    parser.add_argument("datapath", type=to_unicode)
    
    args = parser.parse_args()
    
    ai_sv = "rs://" + args.server_ip + ":" + args.port
    ai_realm = args.realm
    
    with U.single_threaded_session():
        # create a Wamp session object
        session = Component(ComponentConfig(ai_realm, {}))

        # initialize the msgpack serializer
        serializer = MsgPackSerializer()
        
        # use Wamp-over-rawsocket
        runner = ApplicationRunner(ai_sv, ai_realm, serializers=[serializer])
        
        runner.run(session, auto_reconnect=True)
