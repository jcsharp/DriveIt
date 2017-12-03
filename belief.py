# -*- coding: utf-8 -*-
"""
Belief tracking for the DriveIt Gym environment.
@author: Jean-Claude Manoli
"""
import numpy as np
from gym import spaces
from DriveItMultiGym import DriveItEnv
from car import Car
from PositionTracking import PositionTracking
from filter import LowPassFilter
from DriveItCircuit import * #pylint: disable=W0401,W0614
#pylint: disable=C0301


class BeliefDriveItEnv(DriveItEnv):
    def __init__(self, car=Car(), bots=None, time_limit=10, noisy=True, normalize=True):
        super().__init__(car, bots, time_limit, noisy)
        other_cars = None if bots is None else [b.car for b in bots]
        self.tracker = BeliefTracking(car, other_cars, normalize)
        self.observation_space = self.tracker.observation_space

    def _reset(self, random_position=True):
        obs = super()._reset(random_position)
        return self.tracker.reset(obs)

    def _step(self, action):
        obs, reward, done, info = super()._step(action)
        bel = self.tracker.update(action, obs, self.dt)
        return bel, reward, done, info


class BeliefTracking(PositionTracking):
    look_ahead_time = 0.33
    look_ahead_points = 10
    filter_gain = 0.85

    def __init__(self, car, other_cars, normalize=True):
        super().__init__(car)
        self.other_cars = [] if other_cars is None else other_cars
        self.normalize = normalize
        # x_m, y_m, theta_m, v, k, k_t, k_a
        high = [  checkpoint_median_length,  half_track_width,  pi, car.specs.v_max,  car.specs.K_max,  max_curvature,  max_curvature ]
        low  = [ -checkpoint_median_length, -half_track_width, -pi,             0.0, -car.specs.K_max, -max_curvature, -max_curvature ]
        self.sensor_index = len(low)
        for s in car.dist_sensors:
            high.append(s.specs[0])
            low.append(0.0)
        high, low = np.array(high), np.array(low)
        self._high = high
        if normalize:
            low  = low / high
            high = high / high
        self.observation_space = spaces.Box(low, high)
        self.belief = np.zeros(low.shape, low.dtype)
        self.df = LowPassFilter(self.filter_gain, self.belief[self.sensor_index:])

    def _read_sensors(self):
        for i in range(len(self.car.dist_sensors)):
            self.belief[i + self.sensor_index] = self.car.dist_sensors[i].read(self.other_cars)
        self.belief[self.sensor_index:] = self.df.filter(self.belief[self.sensor_index:])

    def _augment_pos(self, pos):
        x_m, y_m, theta_m, v, k = pos
        k_t = track_curvature(x_m, y_m)
        lhdist = v * self.look_ahead_time * cos(theta_m)
        k_a = curve_ahead(x_m, y_m, lhdist, self.look_ahead_points)
        self._read_sensors()
        self.belief[:self.sensor_index] = x_m, y_m, theta_m, v, k, k_t, k_a
        if self.normalize:
            return self.belief / self._high
        else:
            return self.belief
        
    def reset(self, observation):
        pos = super().reset(observation)
        bel = self._augment_pos(pos)
        LowPassFilter(self.filter_gain, bel[self.sensor_index:])
        return bel

    def update(self, action, observation, dt):
        pos = super().update(action, observation, dt)
        return self._augment_pos(pos)
