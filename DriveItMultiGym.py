# -*- coding: utf-8 -*-
"""
Drive-It competition simulation environment
@author: Jean-Claude Manoli
"""

from os import path
import math
import numpy as np
from numpy import cos, sin, pi, clip #pylint: disable=W0611
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
from car import Car, steer_actions
from utils import * #pylint: disable=W0401,W0614
from DriveItCircuit import * #pylint: disable=W0401,W0614

blue_threshold = 0.9
fps = 60.0
dt = 1.0 / fps

out_reward = -checkpoint_median_length
throttle_override_reward = -dt

max_compass_bias = 0.017
compass_deviation = max_compass_bias / 100.0
velocity_deviation = 0.001


class DriveItEnvMulti(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': fps
    }


    def __init__(self, cars=(Car()), time_limit=10, noisy=True, random_position=True):
        self.cars = cars
        self.time_limit = time_limit
        self.noisy = noisy
        self.random_position = random_position
        self.car_num = len(cars)
        self.dt = dt
        
        self.viewer = None

        high = np.array([ 1.0,  0.0 * pi      , cars[0].specs.v_max,  cars[0].specs.K_max, 1.0 ], dtype=np.float32)
        low  = np.array([ 0.0, -3.0 * pi / 2.0,                 0.0, -cars[0].specs.K_max, 0.0 ], dtype=np.float32)
        self.action_space = spaces.Discrete(len(steer_actions))
        self.observation_space = spaces.Box(low, high)

        fname = path.join(path.dirname(__file__), "track.png")
        self._init_track(fname, width=2.0, height=2.0)

        self._seed()
        self.time = -1.0
        self.observations = {}
        self.state = {}


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        for car in self.cars:
            car.set_noise(self.noisy, self.np_random)
        return [seed]


    def _reset_car(self, i):
        car = self.cars[i]

        if self.random_position:
            # random position along the track median
            x_m = self.np_random.uniform(-checkpoint_median_length, checkpoint_median_length)
            # keep 10 cm distance between cars
            for j in range(i):
                d = abs(x_m - self.state[self.cars[j]][0])
                if d < 0.1 + car.length / 2.0:
                    return self._reset_car(i)
        else:
            space = lap_median_length / len(self.cars)
            x_m = wrap(i * space, -checkpoint_median_length, checkpoint_median_length) - car.length / 2.0

        y_m = 0.0
        theta, K = median_properties(x_m)
        steer = np.round(K / car.specs.K_max / car.specs.steer_step) * car.specs.steer_step

        if self.noisy:
            y_m += self.np_random.uniform(-0.01, 0.01)
            theta += self.np_random.uniform(-pi / 36.0, pi / 36.0)
            steer += self.np_random.choice((-1, 0, 1)) * car.specs.steer_step

        throttle = self.np_random.uniform(0.0, car.safe_throttle(steer)) \
            if self.noisy else car.safe_throttle(steer)
        throttle = np.round(throttle / car.specs.throttle_step) * car.specs.throttle_step

        x, y, _ = median_to_cartesian(x_m, y_m, 0.0)

        return x_m, car.reset(x, y, theta, steer, throttle)
        

    def _reset(self):
        '''
        Resets the simulation.
        '''

        self.time = 0.0
        self.observations = {}
        self.state = {}

        for i in range(len(self.cars)):
            x_m, (x, y, theta, _, _, v, K) = self._reset_car(i)
            car = self.cars[i]
            if self.noisy:
                compass_bias = self.np_random.uniform(-max_compass_bias, max_compass_bias)
                theta -= compass_bias
            else:
                compass_bias = 0.0
            blue = self._blueness(x, y)
            checkpoint = 1.0 if x_m < 0.0 else 0.0
            self.observations[car] = blue, theta, v, K, checkpoint
            self.state[car] = x_m, compass_bias, 0 #laps
        
        return self.observations


    def _step(self, actions):

        if len(actions) != self.car_num:
            raise ValueError('Wrong number of actions.')

        self.time += dt
        rewards = {}
        exits = []
        info = {}
        done = False

        for car in self.cars:
            action = actions[car]
            x_m_, compass_bias, laps = self.state[car]

            # move the car
            x, y, theta, _, _, v, K_hat, throttle_override = car.step(action, dt)

            # read sensors
            blue = self._blueness(x, y)

            if self.noisy:
                # add observation noise
                compass_bias = clip(self.np_random.normal(compass_bias, compass_deviation), -max_compass_bias, max_compass_bias)
                theta_hat = theta + compass_bias
                v_noise = 0.0 if v <= 0 else self.np_random.normal(0, v * velocity_deviation)
                v_hat = v + v_noise
            else:
                theta_hat = theta
                v_hat = v

            # check progress along the track
            x_m, y_m, _ = cartesian_to_median(x, y, theta)
            lap_threshold = x_m > 0.0 and x_m_ <= 0.0
            checkpoint_threshold = x_m < 0.0 and x_m_ > 0.0
            checkpoint = 1.0 if x_m < 0.0 else 0.0
            dx_m = x_m - x_m_
            if lap_threshold:
                laps += 1
                info['threshold'] = 'lap'
            if checkpoint_threshold:
                dx_m += lap_median_length
                info['threshold'] = 'checkpoint'

            # reward progress
            reward = dx_m
            if throttle_override > 0.0:
                reward += throttle_override_reward * throttle_override
            if abs(y_m) - half_track_width > 0.0:
                reward = out_reward
                done = True
                info['done'] = 'out'
                exits.append(car)

            self.observations[car] = blue, theta_hat, v_hat, K_hat, checkpoint
            rewards[car] = reward
            self.state[car] = x_m, compass_bias, laps

        # are we done yet?
        if self.time >= self.time_limit:
            done = True
            info['done'] = 'complete'

        # collision detection
        if self.car_num > 1:
            for car1 in self.cars:
                car2 = car1.detect_collision(self.cars)
                if car2 is not None:
                    rewards[car1] = out_reward
                    rewards[car2] = out_reward
                    # if self.state[car1][0] < 0 and self.state[car2][0] > 0:
                    #     rewards[car2] = out_reward
                    # else:
                    #     rewards[car1] = out_reward
                    done = True
                    info['done'] = 'crash'

        if done:
            info['laps'] = laps

        return self.observations, rewards, done, info


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


    def _init_track(self, fname, width: float, height: float):
        self.track = rendering.Image(fname, width, height)
        self.img_offset = width / 2.0, height / 2.0
        self.img_scale = self.track.img.width / width, self.track.img.height / height


    def _track_to_image_coordinates(self, x, y):
        ox, oy = self.img_offset
        scalex, scaley = self.img_scale
        img_x = math.trunc((x + ox) * scalex)
        img_y = math.trunc((oy - y) * scaley)
        return img_x, img_y

    def _track_color(self, x, y):
        img_x, img_y = self._track_to_image_coordinates(x, y)
        return self._img_color(img_x, img_y)

    def _track_color_average(self, x, y, n=4):
        '''
        Gets the track color at the specified coordinates, averaging the pixels around that position.
        '''
        img_x, img_y = self._track_to_image_coordinates(x, y)
      
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
        b, _, r, _ = self._track_color(x, y)
        return (b - r) / 217



class DriveItEnv(DriveItEnvMulti):

    def __init__(self, car=Car(), bots=None, time_limit=10, noisy=True, random_position=True):
        self.car = car
        self.bots = [] if bots is None else bots
        cars = []
        cars.append(car)
        for bot in self.bots:
            cars.append(bot.car)
        super().__init__(cars, time_limit, noisy, random_position)

    def _reset(self):
        obs = super()._reset()
        for i in range(1, self.car_num):
            self.bots[i-1].reset(self.state[self.cars[i]][0], obs[self.cars[i]])
        return obs[self.car]

    def _step(self, action):
        actions = {}
        actions[self.car] = action
        for i in range(1, self.car_num):
            actions[self.cars[i]] = self.bots[i-1].act()

        obs, rewards, done, info = super()._step(actions)
        
        for i in range(1, self.car_num):
            self.bots[i-1].observe(obs[self.cars[i]], dt)

        return obs[self.car], rewards[self.car], done, info
