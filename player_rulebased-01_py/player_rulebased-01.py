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
            self.cur_ball = [0, 0]
            self.prev_ball = [0, 0]
            self.idx = 0
            self.wheels = [0 for _ in range(10)]

            self.distances = [i for i in range(5)]
            self.idxs = [i for i in range(5)]
            self.deadlock_cnt = 0
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

    def find_closest_robot_2(self):
        for i in range(self.number_of_robots):
            self.distances[i] = helper.distance(self.cur_ball[X], self.cur_my_posture[i][X], self.cur_ball[Y], self.cur_my_posture[i][Y])
        self.idxs = sorted(range(len(self.distances)), key=lambda k: self.distances[k])

    def count_deadlock(self):
        d_ball = helper.distance(self.cur_ball[X], self.prev_ball[X], self.cur_ball[Y], self.prev_ball[Y]) # delta of ball
        #self.printConsole("boal delta: " + str(d_ball))
        if (abs(self.cur_ball[Y]) > 0.65) and (d_ball < 0.02):
            #self.printConsole("boal stop")
            self.deadlock_cnt += 1
        else:
            self.deadlock_cnt = 0

    def count_goal_area(self):
        count = 0
        for i in range(self.number_of_robots):
            if (abs(self.cur_my_posture[i][X]) > 1.6) and (abs(self.cur_my_posture[i][Y]) < 0.43):
                count += 1
        return count

    def count_penalty_area(self):
        count = 0
        for i in range(self.number_of_robots):
            if (abs(self.cur_my_posture[i][X]) > 1.3) and (abs(self.cur_my_posture[i][Y]) < 0.7):
                count += 1
        return count        

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

            if (ball_dist < 0.04):
                # if near the ball and near the opposite team goal
                if (goal_dist < 1.0):
                    self.position(robot_id, self.field[X]/2, 0)
                    self.printConsole("shoot")
                else:
                    # if near and in front of the ball
                    if (self.cur_ball[X] < self.cur_my_posture[robot_id][X] - 0.044):
                        x_suggest = self.cur_ball[X] - 0.044
                        self.position(robot_id, x_suggest, self.cur_ball[Y])
                        self.printConsole("    near front")
                    # if near and behind the ball
                    else:
                        self.position(robot_id, self.field[X] + self.goal[X], -self.goal[Y]/2)
                        self.printConsole("        near behind")
            else:
                if (self.cur_ball[X] < self.cur_my_posture[robot_id][X]):
                    if (self.cur_ball[Y] > 0):
                        self.position(robot_id, self.cur_ball[X] - ox, min(self.cur_ball[Y] - oy, 1.2))
                        #self.printConsole("            far front +")
                    else:
                        self.position(robot_id, self.cur_ball[X] - ox, max(self.cur_ball[Y] + oy, -1.2))
                        #self.printConsole("            far front -")
                else:
                    if (self.cur_ball[Y] > 0):
                        self.position(robot_id, self.cur_ball[X], min(self.cur_ball[Y], 2))
                        #self.printConsole("                far behind +")
                    else:
                        self.position(robot_id, self.cur_ball[X], max(self.cur_ball[Y], -2))
                        #self.printConsole("                far behind -")
##############################################################################
        # new rule
        def chase(self, robot_id, x, y):
            self.position(robot_id, x, y)

        def avoid_deadlock(self):
            self.position(0, self.cur_ball[X], 0)
            self.position(1, self.cur_ball[X], 0)
            self.position(2, self.cur_ball[X], 0)
            self.position(3, self.cur_ball[X], 0)
            self.position(4, self.cur_ball[X], 0)
        
        def avoid_goal_foul(self):
            midfielder(self, self.idxs[0], self.idxs[0], 0)
            chase(self, self.idxs[1], self.cur_my_posture[self.idxs[0]][X], self.cur_my_posture[self.idxs[0]][Y])
            self.position(self.idxs[2], 0, 0)
            self.position(self.idxs[3], 0, 0)
            self.position(self.idxs[4], 0, 0)

        def avoid_penalty_foul(self):
            midfielder(self, self.idxs[0], self.idxs[0], 0)
            chase(self, self.idxs[1], self.cur_my_posture[self.idxs[0]][X], self.cur_my_posture[self.idxs[0]][Y])
            chase(self, self.idxs[2], self.cur_my_posture[self.idxs[1]][X], self.cur_my_posture[self.idxs[1]][Y])
            self.position(self.idxs[3], 0, 0)
            self.position(self.idxs[4], 0, 0)            

##############################################################################                
            
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
            self.get_coord(received_frame)
            self.find_closest_robot_2()
            self.count_deadlock()
            goal_area_count = self.count_goal_area()
            penalty_area_count = self.count_penalty_area()

            warning = (self.deadlock_cnt > 30) or (goal_area_count > 2) or (penalty_area_count > 3)

            self.printConsole("goal area count: " + str(goal_area_count))
            self.printConsole("    penalty area count: " + str(penalty_area_count))
            # self.printConsole("robot 0 active: " + str(self.cur_my_posture[0][ACTIVE]))
            # self.printConsole("robot 1 active: " + str(self.cur_my_posture[1][ACTIVE]))
            # self.printConsole("robot 2 active: " + str(self.cur_my_posture[2][ACTIVE]))
            # self.printConsole("robot 3 active: " + str(self.cur_my_posture[3][ACTIVE]))
            # self.printConsole("robot 4 active: " + str(self.cur_my_posture[4][ACTIVE]))
            # self.printConsole("robot " + str(self.idxs[0]) + " distance: " + str(self.distances[self.idxs[0]]))
            # self.printConsole("  robot " + str(self.idxs[1]) + " distance: " + str(self.distances[self.idxs[1]]))
            # self.printConsole("    robot " + str(self.idxs[2]) + " distance: " + str(self.distances[self.idxs[2]]))
            # self.printConsole("      robot " + str(self.idxs[3]) + " distance: " + str(self.distances[self.idxs[3]]))
            # self.printConsole("        robot " + str(self.idxs[4]) + " distance: " + str(self.distances[self.idxs[4]]))
        
            if goal_area_count > 2:
                self.printConsole("                                                     goal foul!")
                avoid_goal_foul(self)
            elif penalty_area_count > 3:
                self.printConsole("                                                     penalty foul!")
                avoid_penalty_foul(self)
            elif self.deadlock_cnt > 25:
                self.printConsole("                        warning: deadlock")
                avoid_deadlock(self)                
            else:
                midfielder(self, self.idxs[0], self.idxs[0], 0)
                chase(self, self.idxs[1], self.cur_my_posture[self.idxs[0]][X], self.cur_my_posture[self.idxs[0]][Y])
                chase(self, self.idxs[2], self.cur_my_posture[self.idxs[1]][X], self.cur_my_posture[self.idxs[1]][Y])
                chase(self, self.idxs[3], self.cur_my_posture[self.idxs[2]][X], self.cur_my_posture[self.idxs[2]][Y])
                chase(self, self.idxs[4], self.cur_my_posture[self.idxs[3]][X], self.cur_my_posture[self.idxs[3]][Y])

            #goalie(self, 4)
            #defender(self, 3, self.idx, 0.2)
            #defender(self, 2, self.idx, -0.2)
            #midfielder(self, 1, self.idx, 0.15)
            #midfielder(self, 0, self.idx, -0.15)

            self.prev_ball = self.cur_ball
            set_wheel(self, self.wheels)   
            #self.printConsole("goal: " + str(self.goal[X]) + ", " + str(self.goal[Y]))
            #self.printConsole("field: " + str(self.field[X]) + ", " + str(self.field[Y]))      
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
