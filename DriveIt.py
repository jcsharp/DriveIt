# -*- coding: utf-8 -*-
"""
Drive-It competition simulation environment
@author: Jean-Claude Manoli
"""

import math
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np
from numpy import abs, cos, sin, pi
from os import path

class DriveItEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    steer_actions = [0.0, 0.5, -0.5]
    time_limit = 10.0
    blue_threshold = 0.75
    fps = 60.0
    dt = 1.0 / fps

    def __init__(self):

        self.viewer = None

        high = np.array([pi, 6.0, 3.0,  1.0,  1.0])
        low = np.array([-pi, 0.0, 0.0, -1.0, -1.0])

        self.action_space = spaces.Discrete(len(self.steer_actions)) 
        # for a continuous action space:
        # self.action_space = spaces.Box(self.min_action, self.max_action, shape = (1,)) 
        self.observation_space = spaces.Box(low, high)

        fname = path.join(path.dirname(__file__), "track.png")
        self.track = rendering.Image(fname, 2., 2.)

        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.state = (0.0, 0.0, 0.0, 0.0, 0.0)
        self.hiddenState = (0.0, -0.42, 0.0, 0.0, -1, True, 0.0, 0.0, 0.0, 0.0)
        return np.array(self.state)

    def _step(self, action):

        t, x, y, throttle, lap_count, checkpoint_passed, bl_, br_, lap_timestamp, checkpoint_time \
        = self.hiddenState
        theta, lap_distance, checkpoint_distance, steer, brbl = self.state

        t += self.dt
        lap_time = t - lap_timestamp

        ds = self.steer_actions[action]
        steer_ = max(min(steer + ds, 1.0), -1.0)

        dp = self.auto_throttle(steer_, throttle)
        throttle_ = max(min(throttle + dp, 1.0), -1.0)
        
        # calculate new position
        K = 4.5 * (steer + steer_) / 2.0
        v = 2.5 * (throttle + throttle_) / 2.0
        dd = v * self.dt
        theta = angle_normalize(theta + dd * K)
        cth = cos(theta)
        sth = sin(theta)
        x += dd * cth
        y += dd * sth
        steer = steer_
        throttle = throttle_

        lap_distance += dd
        if checkpoint_passed: 
            checkpoint_distance += dd

        blueness = self._blueness(x,y)
        sens_x = 0.11
        sens_y = 0.05
        offset_x_l = sens_x * cth - sens_y * sth
        offset_x_r = sens_x * cth + sens_y * sth
        offset_y_l = sens_x * sth + sens_y * cth
        offset_y_r = sens_x * sth - sens_y * cth
        blue_left = self._blueness(x + offset_x_l, y + offset_y_l)
        blue_right = self._blueness(x + offset_x_r, y + offset_y_r)

        out = blueness >= self.blue_threshold \
            or self.is_wrongWay(x, y, theta, checkpoint_distance)
        timeout = t >= self.time_limit
        done = out or timeout

        # staying alive bonus
        reward = 3 # self.dt
        # encourage going straight
        reward -= abs(steer)
        # encourage steady steering
        # reward -= self.dt * ds if ds * steer >= 0 else 0
        # penalty for getting closer to the track boundary
        reward -= (blue_left - bl_ + blue_right - br_) / 2.0 #* self.dt
        #reward -= (blue_left + blue_right) * self.dt / 2.0
        # scale down the reward
        reward *= self.dt
        # do we need further punishment when we exit the tracks?
        if out: reward = -10.0
        
        lap = checkpoint_passed and self.is_lap(x, theta, dd)
        if lap:
            lap_count += 1
            lap_time = t - lap_timestamp
            lap_timestamp = t
            if lap_count > 0:
                reward += 5.0 \
                - lap_distance + checkpoint_distance \
                #- lap_time + checkpoint_time
            else:
                lap = False
            lap_distance = 0
            checkpoint_distance = 0
            checkpoint_passed = False
        
        checkpoint = (not checkpoint_passed) and self.is_checkpoint(y, theta, dd)
        if checkpoint:
            checkpoint_time = t - lap_timestamp
            reward += 5.0 - lap_distance #- checkpoint_time
            checkpoint_passed = True


        self.state = (theta, lap_distance, checkpoint_distance, steer, blue_right - blue_left)
        self.hiddenState = (t, x, y, throttle, lap_count, checkpoint_passed, blue_left, blue_right, lap_timestamp, checkpoint_time)
        return np.array(self.state), reward, done, { \
            'checkpoint': checkpoint, 
            'lap': lap, 
            'lap_count': lap_count,
            'lap_time': lap_time,
            'checkpoint_time': checkpoint_time,
        }

    def is_lap(self, x, theta, dd):
        if abs(theta) > pi / 2.0: return False
        x_ = x - dd * cos(theta) + 0.105
        return (x_ <= 0.0) and (x_ > - dd)

    def is_checkpoint(self, y, theta, dd):
        if theta < 0.0: return False
        y_ = y - dd * sin(theta) + 0.105
        return (y_ <= 0.0) and (y_ > - dd)

    wrongWayThreshold = 0.225
    wrongWayMargin = 0.275
    
    def is_wrongWay(self, x, y, theta, checkpoint):
        if abs(x) < self.wrongWayThreshold:
            if y > self.wrongWayMargin and theta > 0:
                return not checkpoint
            if y < -self.wrongWayMargin and theta < 0:
                return True
        elif abs(y) < self.wrongWayThreshold:
            if x > self.wrongWayMargin and abs(theta) < pi / 2.0:
                return checkpoint
            if x < -self.wrongWayMargin and abs(theta) > pi / 2.0:
                return True
        
    def auto_throttle(self, steer, throttle):
        safe = 1.0 # - abs(steer) / 3.0
        return math.copysign(self.dt * 2.5, safe - throttle)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            screen_width = 600
            screen_height = 600

            carwidth = 0.12
            carlenght = 0.24
            flcw = 0.02

            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(-1.1, 1.1, -1.1, 1.1)

            self.track.set_color(128,128,128)
            self.viewer.add_geom(self.track)

            l,r,t,b = -carlenght / 2, carlenght / 2, carwidth / 2, -carwidth / 2
            
            self.cartrans = rendering.Transform()

            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.set_color(255, 64, 128)
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            
            carout = rendering.PolyLine([(l,b), (l,t), (r,t), (r,b)], close=True)
            carout.set_linewidth(3)
            carout.set_color(0, 0, 0)
            carout.add_attr(self.cartrans)
            self.viewer.add_geom(carout)
     
            #self.floorWin = rendering.PolyLine([(-flcw,-flcw), (-flcw,flcw), (flcw,flcw), (flcw,-flcw)], close=True)
            #self.floorWin.set_linewidth(3)
            #self.floorWin.set_color(255,0,0)
            #self.floorWin.add_attr(self.cartrans)
            #self.viewer.add_geom(self.floorWin)

        x = self.hiddenState[1]
        y = self.hiddenState[2]
        theta = self.state[0]
        self.cartrans.set_translation(x, y)
        self.cartrans.set_rotation(theta)
        #b,g,r,a = self._track_color(x,y)
        #self.floorWin.set_color(r,g,b)

        return self.viewer.render(return_rgb_array = mode == 'rgb_array')

    def _track_color(self, x, y):
        xx = math.trunc((x + self.track.width / 2) / self.track.width * self.track.img.width)
        yy = math.trunc((self.track.height - (y + self.track.height / 2)) / self.track.height * self.track.img.height)
        pos = xx * 4 + yy * self.track.img.width * 4
        if pos < 0 or pos > len(self.track.img.data):
            return (0,0,0,0)
        else: 
            return self.track.img.data[pos:pos + 4]

    def _blueness(self, x, y):
        b,g,r,a = self._track_color(x,y)
        if a == 0: return 1.0
        else: return (b - r) / 255

def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)
