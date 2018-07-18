#!/usr/bin/python3

# keunhyung 7/17
# keras dqn refer to atari breakout
# goalkeeper

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

from aiwc_dqn import DQNAgent

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
    """
    AI Base + Skeleton
    """ 

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
            self.idx = 0
            self.wheels = [0 for _ in range(10)]

            self.state_size = 5 # the number of possible states
            self.action_size = 4 # the number of possible actions
            self.agent = DQNAgent(self.state_size, self.action_size)       
            self.global_step = 0 # iteration step             
            self.save_every_steps = 10000 # save the model

            self.step = 0 # statistic step 
            self.stats_steps = 6000 # for tensorboard
            self.reward_sum = 0
            self.score_op_sum =0
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

    def find_closest_robot(self):
        min_idx = 0
        min_distance = 9999.99
        for i in range(self.number_of_robots-1):
            measured_distance = helper.distance(self.cur_ball[X], self.cur_my_posture[i][X], self.cur_ball[Y], self.cur_my_posture[i][Y])
            if (measured_distance < min_distance):
                min_distance = measured_distance
                min_idx = i
        self.idx = min_idx

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

        def goalie(self, robot_id):
            # Goalie just track the ball[Y] position at a fixed position on the X axis
            x = (-self.field[X]/2) + (self.robot_size/2) + 0.05
            y = max(min(self.cur_ball[Y], (self.goal[Y]/2 - self.robot_size/2)), -self.goal[Y]/2 + self.robot_size/2)
            self.position(robot_id, x, y)
            
        def defender(self, robot_id, idx, offset_y):
            ox = 0.1
            oy = 0.075
            min_x = (-self.field[X]/2) + (self.robot_size/2) + 0.05 
            
            # If ball is on offense
            if (self.cur_ball[X] > 0):
                # If ball is in the upper part of the field (y>0)
                if (self.cur_ball[Y] > 0):
                    self.position(robot_id, 
                                  (self.cur_ball[X]-self.field[X]/2)/2, 
                                  (min(self.cur_ball[Y],self.field[Y]/3))+offset_y)
                # If ball is in the lower part of the field (y<0)
                else:
                    self.position(robot_id, 
                                  (self.cur_ball[X]-self.field[X]/2)/2, 
                                  (max(self.cur_ball[Y],-self.field[Y]/3))+offset_y)
            # If ball is on defense
            else:
                # If robot is in front of the ball
                if (self.cur_my_posture[robot_id][X] > self.cur_ball[X] - ox):
                    # If this defender is the nearest defender from the ball
                    if (robot_id == idx):
                        self.position(robot_id, 
                                      (self.cur_ball[X]-ox), 
                                      ((self.cur_ball[Y]+oy) if (self.cur_my_posture[robot_id][Y]<0) else (self.cur_ball[Y]-oy)))
                    else:
                        self.position(robot_id, 
                                      (max(self.cur_ball[X]-0.03, min_x)), 
                                      ((self.cur_my_posture[robot_id][Y]+0.03) if (self.cur_my_posture[robot_id][Y]<0) else (self.cur_my_posture[robot_id][Y]-0.03)))
                # If robot is behind the ball
                else:
                    if (robot_id == idx):
                        self.position(robot_id, 
                                      self.cur_ball[X], 
                                      self.cur_ball[Y])                        
                    else:
                        self.position(robot_id, 
                                      (max(self.cur_ball[X]-0.03, min_x)), 
                                      ((self.cur_my_posture[robot_id][Y]+0.03) if (self.cur_my_posture[robot_id][Y]<0) else (self.cur_my_posture[robot_id][Y]-0.03)))

        def midfielder(self, robot_id, idx, offset_y):
            ox = 0.1
            oy = 0.075
            ball_dist = helper.distance(self.cur_my_posture[robot_id][X], self.cur_ball[X], self.cur_my_posture[robot_id][Y], self.cur_ball[Y])
            goal_dist = helper.distance(self.cur_my_posture[robot_id][X], self.field[X]/2, self.cur_my_posture[robot_id][Y], 0)
            
            if (robot_id == idx):
                if (ball_dist < 0.04):
                    # if near the ball and near the opposite team goal
                    if (goal_dist < 1.0):
                        self.position(robot_id, self.field[X]/2, 0)
                    else:
                        # if near and in front of the ball
                        if (self.cur_ball[X] < self.cur_my_posture[robot_id][X] - 0.044):
                            x_suggest = self.cur_ball[X] - 0.044
                            self.position(robot_id, x_suggest, self.cur_ball[Y])
                        # if near and behind the ball
                        else:
                            self.position(robot_id, self.field[X] + self.goal[X], -self.goal[Y]/2)
                else:
                    if (self.cur_ball[X] < self.cur_my_posture[robot_id][X]):
                        if (self.cur_ball[Y] > 0):
                            self.position(robot_id, self.cur_ball[X] - ox, min(self.cur_ball[Y] - oy, 0.45*self.field[Y]))
                        else:
                            self.position(robot_id, self.cur_ball[X] - ox, min(self.cur_ball[Y] + oy, -0.45*self.field[Y]))
                    else:
                        self.position(robot_id, self.cur_ball[X], self.cur_ball[Y])
            else:
                self.position(robot_id, self.cur_ball[X]-0.1, self.cur_ball[Y]+offset_y)

        def set_action(robot_id, action_number):
            if action_number == 0:
                self.wheels[2*robot_id] = self.max_linear_velocity
                self.wheels[2*robot_id + 1] = self.max_linear_velocity
                # Go Forward with fixed velocity
            elif action_number == 1:
                self.wheels[2*robot_id] = 0
                self.wheels[2*robot_id + 1] = 0
                # Do not move
            elif action_number == 2:
                self.wheels[2*robot_id] = -self.max_linear_velocity
                self.wheels[2*robot_id + 1] = -self.max_linear_velocity
                # Go Backward with fixed velocity
            elif action_number == 3:
                self.wheels[2*robot_id] = -1
                self.wheels[2*robot_id + 1] = 1
                # Spin
            
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
                        
##############################################################################
            # next state
            self.get_coord(received_frame)
            self.find_closest_robot()
            next_state = [round(self.cur_my_posture[4][X]/2.05, 2), round(self.cur_my_posture[4][Y]/1.35, 2), round(self.cur_my_posture[4][TH]/(2*math.pi), 2), 
                round(self.cur_ball[X]/2.05, 2), round(self.cur_ball[Y]/1.35, 2)]
            next_state = np.reshape([next_state], (1, self.state_size, 1))
            next_history = np.append(next_state, self.agent.history[:, :, :3], axis=2)
            #self.printConsole(next_state)
            # Reward
            reward = 0
            if (self.cur_ball[X] < -1.4) and (-0.65 < self.cur_ball[Y] < 0.65):
                # if the ball is near my goal post -> penalty
                reward += (self.cur_ball[X] + 1.4) + (abs(self.cur_ball[Y]) -0.65)
            if(received_frame.reset_reason == SCORE_OPPONENT):
                # if oppenent goal -> penalty
                reward -= 5
                self.score_op_sum += 1
                self.printConsole("score opponet")
            if reward != 0:
                self.printConsole("step reward: " + str(reward))
            # Reset
            reset = (received_frame.reset_reason == SCORE_MYTEAM) or (received_frame.reset_reason == SCORE_OPPONENT) or (received_frame.reset_reason == DEADLOCK)
          
            # save the sample <s, a, r, s'> to the replay memory
            self.agent.replay_memory(self.agent.history, self.agent.action, reward, next_history, reset)
            # every time step do the training
            if len(self.agent.memory) >= self.agent.train_start:
                self.agent.train_replay()

            # save the history and get action
            self.agent.history = next_history
            self.agent.action = self.agent.get_action(np.reshape(self.agent.history, (1, -1)))
            #self.printConsole("agent action: " + str(self.agent.action))

            # Set goalkeeper wheels
            set_action(4, self.agent.action)
            if (self.cur_my_posture[4][X] > -1.4) or (abs(self.cur_my_posture[4][Y]) > 0.65):
                # if goalkeeper is far from my goal post
                self.position(4, -self.field[X]/2, 0) # go to goal post

            # other robot Functions
            #goalie(self, 4)
            #defender(self, 3, self.idx, 0.2)
            #defender(self, 2, self.idx, -0.2)
            #midfielder(self, 1, self.idx, 0.15)
            #midfielder(self, 0, self.idx, -0.15)

            set_wheel(self, self.wheels)

            # go one step
            self.global_step += 1
            self.step += 1
            self.reward_sum += reward            
##############################################################################   
            # for tensorboard         
            self.agent.avg_q_max += np.amax(
                self.agent.model.predict(np.reshape(self.agent.history, (1, -1)))[0])
##############################################################################
            if self.global_step % 50 == 0:
                # print current status
                self.printConsole("step: " + str(self.global_step) + ", Epsilon: " + str(self.agent.epsilon))
                self.printConsole("agent action: " + str(self.agent.action))

            if self.global_step % self.agent.update_target_rate == 0:
                # every reset update the target model to be same with model
                self.agent.update_target_model()
                self.printConsole("update target")

            if self.global_step % self.save_every_steps == 0: # save the model
                self.agent.save_model("./save_model/aiwc_dqn.h5")
                self.printConsole("Saved model")

            if self.global_step % self.stats_steps == 0: # plot the statics
                stats = [self.reward_sum, self.agent.avg_q_max / float(self.step), self.step,
                             self.agent.avg_loss / float(self.step)]
                for i in range(len(stats)):
                    self.agent.sess.run(self.agent.update_ops[i], feed_dict={
                        self.agent.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.agent.sess.run(self.agent.summary_op)
                self.agent.summary_writer.add_summary(summary_str, self.global_step / self.stats_steps)

                self.printConsole("average reward: " + str(self.reward_sum / float(self.step)) + 
                    ", average_q: " + str(self.agent.avg_q_max / float(self.step)) + 
                    ", average loss: " + str(self.agent.avg_loss / float(self.step)) +
                    ", oppenent socore for " + str(self.stats_steps) + " steps: " + str(self.score_op_sum))
                self.reward_sum, self.agent.avg_q_max, self.agent.avg_loss = 0, 0, 0
                self.score_op_sum, self.step = 0, 0
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
