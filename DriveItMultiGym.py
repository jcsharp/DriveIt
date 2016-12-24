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

        # corresponds to the maximum discounted reward over a median lap
        max_reward = cars[0].specs.v_max * dt / (1 - gamma)
        self.out_reward = -max_reward
        
        self.viewer = None

        high = np.array([  1.0,  1.0,  1.0,  1.0, 1.0, 1.0 ])
        low  = np.array([ -1.0, -1.0, -1.0, -1.0, 0.0, 0.0 ])

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
            x_m = np.random.uniform(-checkpoint_median_length, checkpoint_median_length)
            y_m = np.random.uniform(-0.03, 0.03)
            x, y = self._median_to_cartesian(x_m, y_m)
        
            # keep 10 cm distance between cars
            for j in range(i):
                d, _ = self.cars[j].distance(x, y)
                if d < 0.1 + car.specs.car_lenght / 2.0:
                    return self._reset_car(i, random_position)
        
            theta, K = self._median_properties(x_m)
            steer = K / car.specs.K_max
        
            # add some noise
            theta += np.random.uniform(-pi / 36.0, pi / 36.0)
            steer += np.random.randint(-1, 1) * car.specs.steer_step
            throttle = 0.0 #int(self._safe_throttle(steer) * np.random.uniform() / throttle_step) * throttle_step
        
        else:
            # the default startup position, on the lap threshold
            x, y, theta, steer = -median_radius, 0.0, 0.0, 0.0
            x_m, y_m, blue_left, blue_right = 0.0, 0.0, 0.0, 0.0
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
        self.observations = []
        self.state = []
        placed_cars = []

        for i in range(len(self.cars)):
            x, y, theta, steer, throttle, odometer, v = self._reset_car(i, random_position)
            self.observations.append(np.array((odometer, 0.0, theta, steer, v)))
            self.state.append((odometer, 0.0))
        
        return self.observations


    def _step(self, actions):

        self.time += dt
        observations = []
        rewards = []
        rewards = []
        state = []

        for i in range(len(actions)):
            a = actions[i]
            car = self.cars[i]
            x_m_, bias = self.state[i]

            # move the car
            x, y, theta, steer, throttle, d, v = car.step(a, dt)

            # read sensors
            blue = self._blueness(x, y)

            if self.noisy:
                # add observation noise
                bias = max(-0.017, min(0.017, np.random.normal(bias, 0.001)))
                theta_hat = canonical_angle(theta + bias)
                v_noise = 0.0 if v == 0 else np.random.normal(0, v * 0.003)
                v_hat = v + v_noise
                d += v_noise * dt
            else:
                theta_hat = theta
                v_hat = v

            # check progress along the track
            x_m, lap, checkpoint = self._median_distance(x, y, d)
            y_m = self._lateral_error(x, y, x_m)
            dx_m = x_m - x_m_
            if lap:
                d = 0
                car.state = (x, y, theta, steer, throttle, d, v)
            if checkpoint:
                dx_m += lap_median_length
                d = -checkpoint_median_length
                car.state = (x, y, theta, steer, throttle, d, v)

            # are we done yet?
            out = blue >= blue_threshold
            wrong_way = self._is_wrong_way(x, y, theta, x_m < 0.0)

            # reward progress
            reward = dx_m
            if out or wrong_way:
                reward = self.out_reward

            observations.append((d, blue, theta_hat, steer, v_hat))
            rewards.append(reward)
            state.append((x_m, bias))

        self.state = state
        timeout = self.time >= self.time_limit

        # TODO: collision check
        done = timeout #or out or wrong_way

        #if self.show_belief_state:
        #    self.belief = self.update_belief(observation)
        #    retval = self._normalize_belief(self.belief)
        #else:
        #    retval = self._normalize_observation(observation)

        #self.observation = observation

        return observations, rewards, done, { \
            'done': 'complete' if timeout else 'out' if out else 'wrong way' if wrong_way else 'unknown'
        }


    #def update_belief(self, observation):

    #    x_, y_, checkpoint = self.belief_position
    #    d_, b_, theta_, steer_, v_ = self.observations
    #    d, blueness, theta, steer, v = observation
    #    K_ = K_max * steer_

    #    # update position
    #    a = ((v - v_) / dt + (d - d_) * dt) / 2.0 # average on speed and dist to reduce noise
    #    K_dot = K_max * (steer - steer_) / dt
    #    x, y, _, _, _, _ = self._move(x_, y_, theta_, v_, K_, d_, a, K_dot)

    #    x_m, _, _ = self._median_distance(x, y, d_)
    #    if d == -checkpoint_median_length: # checkpoint
    #        checkpoint = True
    #        x_m = -checkpoint_median_length
    #        y = -median_radius

    #    elif d == 0.0: # lap
    #        checkpoint = False
    #        x_m = 0
    #        x = -median_radius
        
    #    if checkpoint and x_m > 0.0:
    #        x_m = 0.0
    #        x = -median_radius
        
    #    if x_m > checkpoint_median_length:
    #        x_m = checkpoint_median_length
    #        y = -median_radius

    #    # lateral position
    #    y_m = self._lateral_error(x, y, x_m)
    #    #if blueness >= 0.1:
    #    #    # the blue gradient is almost linear...
    #    #    y_b = np.copysign(half_track_width + 0.00 + (blue_width + 0.00) * (blueness - 1), y_m)
    #    #    dy = y_b - y_m
    #    #    y_m += dy / 3.0

    #    self.belief_position = (x, y, checkpoint)

    #    return x_m, y_m, theta, steer, v, blueness

    
    #def _normalize_observation(self, observation):
    #    o_n = ()
    #    for o in observation:
    #        x, b, th, st, v = o
    #        o_n.append(np.array((x / checkpoint_median_length, b, th / pi, st, v / v_max)))
    #    return o_n


    def _median_distance(self, x, y, current_mdist):
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
            if current_mdist > - loop_median_length and current_mdist <= line_length:
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
            alpha = np.arctan2(dy, dx) + pi / 2.0
            return line_length + alpha * median_radius, False, False

        # upper-left loop
        else:
            dy = y - median_radius
            dx = -x - median_radius
            alpha = np.arctan2(dx, dy) + pi / 2.0
            return -loop_median_length + alpha * median_radius, False, False


    def _median_to_cartesian(self, x_m, y_m):
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


    def _median_properties(self, x_m):
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
                return pi / 2.0, 0.0
            # upper-left loop
            else:
                alpha = -x_m / median_radius
                return canonical_angle(-alpha), 1.0 / median_radius


    def _lateral_error(self, x, y, x_m):
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


    def _is_wrong_way(self, x, y, theta, checkpoint):
        '''
        Checks if the car is making an illegal turn at the crossing.
        '''
        if abs(x) < half_track_width:
            if y > wrong_way_min and y < wrong_way_max and theta > 0:
                return not checkpoint
            if y < -wrong_way_min and y > -wrong_way_max and theta < 0:
                return True
        elif abs(y) < half_track_width:
            if x > wrong_way_min and x < wrong_way_max and abs(theta) < pi / 2.0:
                return checkpoint
            if x < -wrong_way_min and x > -wrong_way_max and abs(theta) > pi / 2.0:
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

            self.track.set_color(128, 128, 128)
            self.viewer.add_geom(self.track)

            for c in self.cars:
                c.init_rendering(self.viewer)

        for c in self.cars:
            c.render()

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
        if a == 0:
            return 1.0
        else:
            return (b - r) / 217
