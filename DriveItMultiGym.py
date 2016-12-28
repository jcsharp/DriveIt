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

blue_threshold = 0.9
fps = 60.0
dt = 1.0 / fps

# track metrics
median_radius = 0.375
line_length = 2.0 * median_radius
loop_median_length = 3.0 / 2.0 * pi * median_radius
checkpoint_median_length = line_length + loop_median_length
lap_median_length = 2.0 * checkpoint_median_length
half_track_width = 0.225
blue_width = 0.15
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

        # corresponds to the maximum discounted reward over a median lap
        max_reward = cars[0].specs.v_max * dt / (1 - gamma)
        self.out_reward = -max_reward
        
        self.viewer = None

        high = np.array([  checkpoint_median_length, 1.0,  pi, cars[0].specs.v_max,  cars[0].specs.K_max ])
        low  = np.array([ -checkpoint_median_length, 0.0, -pi,                 0.0, -cars[0].specs.K_max ])
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
            x, y = DriveItEnv.median_to_cartesian(x_m, y_m)
        
            # keep 10 cm distance between cars
            for j in range(i):
                d, _ = self.cars[j].distance(x, y)
                if d < 0.1 + car.length / 2.0:
                    return self._reset_car(i, random_position)
        
            theta, K = DriveItEnv.median_properties(x_m)
            steer = K / car.specs.K_max
        
            if self.noisy:
                theta += self.np_random.uniform(-pi / 36.0, pi / 36.0)
                steer += self.np_random.randint(-1, 1) * car.specs.steer_step
                    
        else:
            space = lap_median_length / len(self.cars)
            x_m = wrap(i * space, -checkpoint_median_length, checkpoint_median_length)
            x, y = DriveItEnv.median_to_cartesian(x_m, 0.0)
            theta, K = DriveItEnv.median_properties(x_m)
            steer = K / car.specs.K_max

        throttle = 0.0 
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
            self.observations[car] = np.array((odometer, 0.0, theta, v, K))
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
            x_m, lap, checkpoint = DriveItEnv.median_distance(x, y, x_m_)
            y_m = DriveItEnv.lateral_error(x, y, x_m)
            dx_m = x_m - x_m_
            if lap:
                d = 0
                car.reset_odometer(d)
            if checkpoint:
                dx_m += lap_median_length
                d = -checkpoint_median_length
                car.reset_odometer(d)

            out = blue >= blue_threshold
            wrong_way = DriveItEnv._is_wrong_way(x, y, theta, x_m < 0.0)
            exits.append('out' if out else 'wrong way' if wrong_way else None)

            # reward progress
            reward = dx_m
            if out or wrong_way:
                reward = self.out_reward

            self.observations[car] = (d, blue, theta_hat, v_hat, K)
            rewards[car] = reward
            self.state[car] = (x_m, bias)

        # are we done yet?
        timeout = self.time >= self.time_limit
        done = timeout #or out or wrong_way

        # collision detection
        for car1 in self.cars:
            car2 = car1.detect_collision(self.cars)
            if car2 is not None:
                if self.state[car1][0] < 0 and self.state[car2][0] > 0:
                    rewards[car2] = self.out_reward
                else:
                    rewards[car1] = self.out_reward
                done = True


        return self.observations, rewards, done, { \
            'done': 'complete' if timeout else 'out' if out else 'wrong way' if wrong_way else 'unknown'
        }


    def median_distance(x:float, y:float, current_mdist:float):
        '''
        Calculates the normalized longitudinal position along the track.
        
        Returns (x_m, lap, checkpoint) where:
        x_m: is the normalized longitudinal position along the track,
        lap: is True if the car just passed the lap threshold
        checkpoint: is True if the car just passed the checkpoint threshold
        '''

        # on central cross
        if abs(x) <= median_radius and abs(y) <= median_radius:

            # lap straight line
            if current_mdist > - loop_median_length and current_mdist <= loop_median_length:
                lap = current_mdist < 0
                return x + median_radius, lap, False

            # checkpoint straight line
            else:
                checkpoint = current_mdist > 0
                return -checkpoint_median_length + y + median_radius, False, checkpoint

        # lower-right loop
        if x > -median_radius and y < median_radius:
            dx = x - median_radius
            dy = -y - median_radius
            alpha = np.arctan2(dy, dx) + right_angle
            return line_length + alpha * median_radius, False, False

        # upper-left loop
        else:
            dy = y - median_radius
            dx = -x - median_radius
            alpha = np.arctan2(dx, dy) + right_angle
            return -loop_median_length + alpha * median_radius, False, False


    def median_to_cartesian(x_m:float, y_m:float):
        '''
        Calculates the cartesian coordinates of a specific position relative to the track median.
        '''
        # before checkpoint
        if x_m >= 0:
            # lap straight line
            if x_m < line_length:
                return x_m - median_radius, y_m
            # lower-right loop
            else:
                alpha = (x_m - line_length) / median_radius
                x = (median_radius + y_m) * (sin(alpha) + 1)
                y = (median_radius + y_m) * (cos(alpha) - 1)
                return x, y

        # after checkpoint
        else:
            # checkpoint straight line
            if x_m < -loop_median_length:
                return -y_m, x_m + checkpoint_median_length - median_radius
            # upper-left loop
            else:
                alpha = -x_m / median_radius
                x = (y_m - median_radius) * (1 + sin(alpha))
                y = (median_radius - y_m) * (1 - cos(alpha))
                return x, y


    def median_properties(x_m:float):
        '''
        Calculates the tangent and curvature of a specific positio on the track median.
        '''
        # before checkpoint
        if x_m >= 0:
            # lap straight line
            if x_m < line_length:
                return 0.0, 0.0
            # lower-right loop
            else:
                alpha = (x_m - line_length) / median_radius
                return canonical_angle(-alpha), -1.0 / median_radius

        # after checkpoint
        else:
            # checkpoint straight line
            if x_m < -loop_median_length:
                return right_angle, 0.0
            # upper-left loop
            else:
                alpha = -x_m / median_radius
                return canonical_angle(-alpha), 1.0 / median_radius


    def lateral_error(x:float, y:float, x_m:float):
        '''
        Calculates the lateral distance between the car center and the track median.
        '''

        # before checkpoint
        if x_m >= 0:
            # lap straight line
            if x_m < line_length:
                y_m = y

            # lower-right loop
            else:
                dx = x - median_radius
                dy = y + median_radius
                y_m = math.sqrt(dx ** 2 + dy ** 2) - median_radius

        # after checkpoint
        else:
            # checkpoint straight line
            if x_m < -loop_median_length:
                y_m = -x

            # upper-left loop
            else:
                dx = x + median_radius
                dy = y - median_radius
                y_m = median_radius - math.sqrt(dx ** 2 + dy ** 2) 

        return y_m


    def _is_wrong_way(x:float, y:float, theta:float, checkpoint:bool):
        '''
        Checks if the car is making an illegal turn at the crossing.
        '''
        if abs(x) < half_track_width:
            if y > wrong_way_min and y < wrong_way_max and theta > 0:
                return not checkpoint
            if y < -wrong_way_min and y > -wrong_way_max and theta < 0:
                return True
        elif abs(y) < half_track_width:
            if x > wrong_way_min and x < wrong_way_max and abs(theta) < right_angle:
                return checkpoint
            if x < -wrong_way_min and x > -wrong_way_max and abs(theta) > right_angle:
                return True


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
        if pos < 0 or pos > len(self.track.img.data) + 4:
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
                if c != None:
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
        return (b - r) / 217
