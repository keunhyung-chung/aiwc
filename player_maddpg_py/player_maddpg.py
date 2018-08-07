#!/usr/bin/python3

# keunhyung 8/8
# maddpg
# chase the ball, single agent robot 0

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
from aiwc_ddpg import DDPGAgent

import maddpg_master.maddpg.common.tf_util as U
from maddpg_master.maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
from args import Argument

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

            self.arglist = Argument()

            # Create agent trainers
            self.obs_shape_n = [3 for i in range(1)]
            self.num_adversaries = 0
            self.num_good = 1
            self.state_dim = 3 # 3*my robots, relative to the ball position
            self.history_size = 4 # frame history size
            self.action_dim = 2 # 2
            self.trainers = get_trainers(self.num_adversaries, self.obs_shape_n, self.action_dim, self.arglist)

            self.agent = DDPGAgent(self.state_dim * self.history_size, self.action_dim, self.max_linear_velocity)       
            self.global_step = 0 # iteration step            
            self.save_every_steps = 12000 # save the model every 10 minutes
 
            self.stats_steps = 6000 # for tensorboard
            self.reward_sum = 0
            self.score_sum = 0 
            self.active_flag = [[False for _ in range(5)], [False for _ in range(5)]]   
            self.inner_step = 0
            self.wheels = np.zeros(self.number_of_robots*2)
            self.history = np.zeros([self.state_dim, self.history_size])
            self.action = np.zeros(self.action_dim)
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

	def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
	    # This model takes as input an observation and returns values of all actions
	    with tf.variable_scope(scope, reuse=reuse):
	        out = input
	        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
	        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
	        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
	        return out

	def get_trainers(num_adversaries, obs_shape_n, action_space, arglist):
	    trainers = []
	    model = mlp_model
	    trainer = MADDPGAgentTrainer
	    for i in range(num_adversaries):
	        trainers.append(trainer(
	            "agent_%d" % i, model, obs_shape_n, action_space, i, arglist,
	            local_q_func=(arglist.adv_policy=='ddpg')))
	    for i in range(num_adversaries, self.num_good):
	        trainers.append(trainer(
	            "agent_%d" % i, model, obs_shape_n, action_space, i, arglist,
	            local_q_func=(arglist.good_policy=='ddpg')))
	    return trainers

    def get_coord(self, received_frame):
        self.cur_ball = received_frame.coordinates[BALL]
        self.cur_my_posture = received_frame.coordinates[MY_TEAM]
        self.cur_op_posture =received_frame.coordinates[OP_TEAM]

    def get_reward(self, reset_reason):
        # pre_potential = helper.dipole_potential(self.pre_ball[X], self.pre_ball[Y], 2.1, 3)
        # cur_potential = helper.dipole_potential(self.cur_ball[X], self.cur_ball[Y], 2.1, 3)
        reward = -helper.distance2(self.cur_ball[X], self.cur_my_posture[0][X], self.cur_ball[Y], self.cur_my_posture[0][Y])
        if self.cur_my_posture[0][TOUCH]:
            reward += 100

        # Add dead lock penalty
        # if(reset_reason == SCORE_MYTEAM):
        #     self.score_sum += 1
        #     reward += 24 # minimum 24
        #     self.printConsole("my team goal")

        # if(reset_reason == SCORE_OPPONENT):
        #     self.score_sum -= 1
        #     reward -= 24 # maxmimum -24
        #     self.printConsole("op team goal")

        self.printConsole("reward: " + str(reward))
        self.pre_ball = self.cur_ball
        return reward

    def set_wheel_velocity(self, robot_id, left_wheel, right_wheel):
        multiplier = 1
        
        if(abs(left_wheel) > self.max_linear_velocity or abs(right_wheel) > self.max_linear_velocity):
            if (abs(left_wheel) > abs(right_wheel)):
                multiplier = self.max_linear_velocity / abs(left_wheel)
            else:
                multiplier = self.max_linear_velocity / abs(right_wheel)
        
        self.wheels[2*robot_id] = left_wheel*multiplier
        self.wheels[2*robot_id + 1] = right_wheel*multiplier

    def position(self, robot_id, x, y):
        damping = 0.35
        mult_lin = 3.5
        mult_ang = 0.4
        ka = 0
        sign = 1
        
        dx = x - self.cur_my_posture[robot_id][X]
        dy = y - self.cur_my_posture[robot_id][Y]
        d_e = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
        desired_th = (math.pi/2) if (dx == 0 and dy == 0) else math.atan2(dy, dx)

        d_th = desired_th - self.cur_my_posture[robot_id][TH] 
        while(d_th > math.pi):
            d_th -= 2*math.pi
        while(d_th < -math.pi):
            d_th += 2*math.pi
            
        if (d_e > 1):
            ka = 17/90
        elif (d_e > 0.5):
            ka = 19/90
        elif (d_e > 0.3):
            ka = 21/90
        elif (d_e > 0.2):
            ka = 23/90
        else:
            ka = 25/90
            
        if (d_th > helper.degree2radian(95)):
            d_th -= math.pi
            sign = -1
        elif (d_th < helper.degree2radian(-95)):
            d_th += math.pi
            sign = -1
            
        if (abs(d_th) > helper.degree2radian(85)):
            self.set_wheel_velocity(robot_id, -mult_ang*d_th, mult_ang*d_th)
        else:
            if (d_e < 5 and abs(d_th) < helper.degree2radian(40)):
                ka = 0.1
            ka *= 4
            self.set_wheel_velocity(robot_id, 
                                    sign * (mult_lin * (1 / (1 + math.exp(-3*d_e)) - damping) - mult_ang * ka * d_th),
                                    sign * (mult_lin * (1 / (1 + math.exp(-3*d_e)) - damping) + mult_ang * ka * d_th))        
            
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
                        
            self.global_step += 1                        
            self.get_coord(received_frame)
##############################################################################
            # Next state
            next_state = [round(self.cur_ball[X], 2)- round(self.cur_my_posture[0][X], 2), 
                round(self.cur_ball[Y], 2) - round(self.cur_my_posture[0][Y], 2), 
                round(self.cur_my_posture[0][TH]/(2*math.pi), 2)]
            next_state = np.reshape([next_state], (self.state_dim, 1))
            next_history = np.append(next_state, self.history[:, :self.history_size-1], axis=1)
            self.printConsole("next history: " + str(next_history))

            # Reward
            reward = self.get_reward(received_frame.reset_reason)

            # Reset
            if(received_frame.reset_reason != NONE):
                reset = True
                self.printConsole("reset reason: " + str(received_frame.reset_reason))
            else:
                reset = False
            
            self.agent.train_agent(np.reshape(self.history, -1), self.action, reward, np.reshape(next_history, -1), reset, self.global_step)

            # save the history and get action
            self.history = next_history
            self.action = self.agent.get_action(np.reshape(self.history, -1))
            self.wheels[:2] = self.action
            self.printConsole("                 action: " + str(self.wheels[:2]))

            set_wheel(self, self.wheels.tolist())

            self.reward_sum += reward            
##############################################################################   
            # for tensorboard         
            # self.agent.avg_q_max += self.agent.critic.predict([np.reshape(self.history, (1, -1)), 
            #     np.reshape(self.agent.action, (1, -1))])[0]
##############################################################################
            # if self.global_step % 50 == 0:
            #     # print current status
            #     self.printConsole("step: " + str(self.global_step) + ", Epsilon: " + str(self.agent.epsilon))

            if self.global_step % self.save_every_steps == 0: # save the model
                self.agent.saver.save(self.agent.sess, self.agent.save_file)
                self.printConsole("Saved model")

            if reset: # plot the statics
                self.printConsole("add data to tensorboard")
                stats = [self.reward_sum, self.agent.avg_q_max / float(self.stats_steps),
                             self.agent.loss_sum / float(self.stats_steps), self.score_sum]
                for i in range(len(stats)):
                    self.agent.sess.run(self.agent.update_ops[i], feed_dict={
                        self.agent.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.agent.sess.run(self.agent.summary_op)
                self.agent.summary_writer.add_summary(summary_str, self.inner_step)

                self.reward_sum, self.agent.avg_q_max, self.agent.loss_sum = 0, 0, 0
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
    
    # create a Wamp session object
    session = Component(ComponentConfig(ai_realm, {}))

    # initialize the msgpack serializer
    serializer = MsgPackSerializer()
    
    # use Wamp-over-rawsocket
    runner = ApplicationRunner(ai_sv, ai_realm, serializers=[serializer])
    
    runner.run(session, auto_reconnect=True)
