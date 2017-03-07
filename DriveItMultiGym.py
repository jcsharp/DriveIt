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
from numpy import cos, sin, pi
from os import path
from car import *
from utils import *
from DriveItCircuit import *

blue_threshold = 0.9
fps = 60.0
dt = 1.0 / fps

wrong_way_min = 0.275
wrong_way_max = median_radius

class DriveItEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': fps
    }


    def __init__(self, cars=(Car()), time_limit=10, gamma=0.98, noisy=True):
        
        self.cars = cars
        self.time_limit = time_limit
        self.noisy = noisy
        self.car_num = len(cars)
        self.dt = dt

        # corresponds to the maximum discounted reward over a median lap
        max_reward = cars[0].specs.v_max * dt / (1 - gamma)
        self.out_reward = -max_reward
        
        self.viewer = None

        high = np.array([  checkpoint_median_length, 1.0,  pi, cars[0].specs.v_max,  cars[0].specs.K_max, 1.0 ])
        low  = np.array([ -checkpoint_median_length, 0.0, -pi,                 0.0, -cars[0].specs.K_max, 0.0 ])
        self.action_space = spaces.Discrete(len(steer_actions))
        self.observation_space = spaces.Box(low, high)

        fname = path.join(path.dirname(__file__), "track.png")
        self.track = rendering.Image(fname, 2., 2.)

        self._seed()
        self.time = -1.0
        self.reset()


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _reset_car(self, i, random_position):

        car = self.cars[i]

        if random_position:
            # random position along the track median
            x_m = self.np_random.uniform(-checkpoint_median_length, checkpoint_median_length)
            y_m = self.np_random.uniform(-0.01, 0.01) if self.noisy else 0.0
            x, y, _ = median_to_cartesian(x_m, y_m, 0.0)
        
            # keep 10 cm distance between cars
            for j in range(i):
                d, _ = self.cars[j].distance(x, y)
                if d < 0.1 + car.length / 2.0:
                    return self._reset_car(i, random_position)
        
            theta, K = median_properties(x_m)
            steer = K / car.specs.K_max
        
            if self.noisy:
                theta += self.np_random.uniform(-pi / 36.0, pi / 36.0)
                steer += self.np_random.randint(-1, 1) * car.specs.steer_step
                    
        else:
            space = lap_median_length / len(self.cars)
            x_m = wrap(i * space, -checkpoint_median_length, checkpoint_median_length)
            x, y, _ = median_to_cartesian(x_m, 0.0, 0.0)
            theta, K = median_properties(x_m)
            steer = K / car.specs.K_max

        throttle = self.np_random.uniform(0.0, Car._safe_throttle(steer)) \
            if self.noisy else Car._safe_throttle(steer)
        return car.reset(x, y, theta, steer, throttle, x_m)
        

    def _reset(self, random_position=True):
        '''
        Resets the simulation.

        By default, the cars are placed at random positions along the race track, 
        which improves learning.

        If random_position is set to False, the cars are placed evenly on the track median.
        '''

        self.time = 0.0
        self.observations = {}
        self.state = {}

        for i in range(len(self.cars)):
            x, y, theta, steer, throttle, odometer, v, K = self._reset_car(i, random_position)
            car = self.cars[i]
            self.observations[car] = np.array((odometer, 0.0, theta, v, K, 1.0 if odometer < 0.0 else 0.0))
            self.state[car] = (odometer, 0.0)
        
        return self.observations


    def _step(self, actions):

        if len(actions) != self.car_num:
            raise ValueError('Wrong number of actions.')

        self.time += dt
        rewards = {}
        exits = []

        for car in self.cars:
            action = actions[car]
            x_m_, bias = self.state[car]

            # move the car
            x, y, theta, steer, throttle, d, v, K = car.step(action, dt)

            # read sensors
            blue = self._blueness(x, y)

            if self.noisy:
                # add observation noise
                bias = max(-0.01, min(0.01, self.np_random.normal(bias, 0.0001)))
                theta_hat = canonical_angle(theta + bias)
                v_noise = 0.0 if v <= 0 else self.np_random.normal(0, v * 0.003)
                v_hat = v + v_noise
                d += v_noise * dt
            else:
                theta_hat = theta
                v_hat = v

            # check progress along the track
            x_m, lap, checkpoint = median_distance(x, y, x_m_)
            y_m = lateral_error(x, y, x_m)
            dx_m = x_m - x_m_
            if lap:
                d = 0
                car.reset_odometer(d)
            if checkpoint:
                dx_m += lap_median_length
                d = -checkpoint_median_length
                car.reset_odometer(d)

            # reward progress
            reward = dx_m
            if (abs(y_m) > half_track_width) or (blue > blue_threshold):
                reward = self.out_reward
                exits.append(car)

            self.observations[car] = (d, blue, theta_hat, v_hat, K, 1.0 if x_m < 0.0 else 0.0)
            rewards[car] = reward
            self.state[car] = (x_m, bias)

        # are we done yet?
        timeout = self.time >= self.time_limit
        done = timeout or len(exits) > 0 #or out or wrong_way

        # collision detection
        if self.car_num > 1:
            for car1 in self.cars:
                car2 = car1.detect_collision(self.cars)
                if car2 is not None:
                    if self.state[car1][0] < 0 and self.state[car2][0] > 0:
                        rewards[car2] = self.out_reward
                    else:
                        rewards[car1] = self.out_reward
                    done = True


        return self.observations, rewards, done, { \
            'done': 'complete' if timeout else 'out' 
        }


    def _render(self, mode='human', close=False):
        '''
        Draws the track and the car.
        '''
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            screen_width = 600
            screen_height = 600

            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(-1.1, 1.1, -1.1, 1.1)

            self.track.set_color(1, 1, 1)
            self.viewer.add_geom(self.track)

            for c in self.cars:
                c.init_rendering(self.viewer)

        for c in self.cars:
            c.render(self.viewer)

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))


    def _img_color(self, img_x, img_y):
        pos = img_x * 4 + img_y * self.track.img.width * 4
        if pos < 0 or pos > len(self.track.img.data) - 4:
            return None
        else:
            return self.track.img.data[pos:pos + 4]


    def _track_color(self, x, y, n=0):
        '''
        Gets the track color at the specified coordinates, averaging e pixels around that position.
        '''
        img_x = math.trunc((x + self.track.width / 2) / self.track.width * self.track.img.width)
        img_y = math.trunc((self.track.height - (y + self.track.height / 2)) \
            / self.track.height * self.track.img.height)
        
        count = 0
        b, g, r, a = 0, 0, 0, 0
        for i in range(img_x - n, img_x + n + 1):
            for j in range(img_y - n, img_y + n + 1):
                c = self._img_color(i, j)
                if c is not None:
                    b_, g_, r_, a_ = c
                    b += b_; g += g_; r += r_; a += a_
                    count += 1

        if count == 0:
            return (0, 0, 0, 0)
        else:
            return (b / count, g / count, r / count, a / count)


    def _blueness(self, x, y):
        '''
        Gets the blueness of the track at the specified coordinates.

        The blueness is the normalized difference between the blue and the red 
        channels of the (simulated) RGB color sensor.
        '''
        b, g, r, a = self._track_color(x, y, n=1)
        if a == 0:
            return 1.0
        else:
            return (b - r) / 217
