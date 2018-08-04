#!/usr/bin/python3

# keunhyung 7/31
# keras ddgp
# treat all robot as single agent

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
            self.wheels = [0 for _ in range(10)]

            # self.set_seed(0)
            random.seed(0)
            np.random.seed(0)
            tf.set_random_seed(0)

            self.state_dim = 3 # #3*my robots + 2*op robots + 2
            self.action_dim = 2 # 2
            self.agent = DDPGAgent(self.state_dim, self.action_dim, self.max_linear_velocity, -self.max_linear_velocity)
            # self.printConsole("max velocity: " + str(self.max_linear_velocity))       
            self.global_step = 0 # iteration step            
            self.save_every_steps = 12000 # save the model every 10 minutes
 
            self.stats_steps = 6000 # for tensorboard
            self.reward_sum = 0
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

    def get_coord(self, received_frame):
        self.cur_ball = received_frame.coordinates[BALL]
        self.cur_my_posture = received_frame.coordinates[MY_TEAM]
        self.cur_op_posture =received_frame.coordinates[OP_TEAM]

    def get_reward(self, reset_reason):
        # pre_potential = helper.dipole_potential(self.pre_ball[X], self.pre_ball[Y], 2.1, 3)
        # cur_potential = helper.dipole_potential(self.cur_ball[X], self.cur_ball[Y], 2.1, 3)
        reward = helper.dipole_potential(self.cur_my_posture[4][X], self.cur_my_posture[4][Y], 2.1, 0.3)

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
            next_state = [round(self.cur_my_posture[4][X]/2.05, 2), round(self.cur_my_posture[4][Y]/1.35, 2), round(self.cur_my_posture[4][TH]/(2*math.pi), 2)]
            next_state = np.reshape([next_state], (1, self.state_dim, 1))
            next_history = np.append(next_state, self.agent.history[:, :, :3], axis=2)

            # Reward
            reward = self.get_reward(received_frame.reset_reason)

            # Reset
            if(received_frame.reset_reason != NONE):
                reset = True
                self.printConsole("reset reason: " + str(received_frame.reset_reason))
            else:
                reset = False
            
            self.agent.train_agent(np.reshape(self.agent.history, -1), self.wheels[8:10], reward, np.reshape(next_history, -1), reset, self.global_step)

            # save the history and get action
            self.agent.history = next_history
            action = self.agent.get_action(np.reshape(self.agent.history, -1), self.global_step)

            # Set wheels
            self.wheels[8] = action[0]
            self.wheels[9] = action[1]
            # self.printConsole("left wheel: " + str(self.wheels[8]) + "right wheel: " +str(self.wheels[9]))

            self.printConsole("wheels: " + str(self.wheels[8:10]))
            set_wheel(self, self.wheels)

            self.reward_sum += reward            
##############################################################################   
            # for tensorboard         
            # self.agent.avg_q_max += self.agent.critic.predict([np.reshape(self.agent.history, (1, -1)), 
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
