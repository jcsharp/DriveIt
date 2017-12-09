# -*- coding: utf-8 -*-
"""
Belief tracking for the DriveIt Gym environment.
@author: Jean-Claude Manoli
"""
import numpy as np
from gym import spaces
from DriveItMultiGym import DriveItEnv
from car import Car
from PositionTracking import PositionTracking, TruePosition
from filter import LowPassFilter, MovingAverage
from DriveItCircuit import * #pylint: disable=W0401,W0614
#pylint: disable=C0301


class BeliefDriveItEnv(DriveItEnv):
    def __init__(self, car=Car(), bots=None, time_limit=10, noisy=True, random_position=True, max_speed_deviation=0.0, normalize=True):
        super().__init__(car, bots, time_limit, noisy, random_position, max_speed_deviationn)
        other_cars = None if bots is None else [b.car for b in bots]
        if noisy:
            self.tracker = BeliefTracking(car, other_cars, PositionTracking, normalize)
        else:
            self.tracker = BeliefTracking(car, other_cars, TruePosition, normalize)
        self.observation_space = self.tracker.observation_space

    def _reset(self):
        obs = super()._reset()
        x_m = self.state[self.car][0]
        return self.tracker.reset(x_m, obs)

    def _step(self, action):
        obs, reward, done, info = super()._step(action)
        bel = self.tracker.update(action, obs, self.dt)
        return bel, reward, done, info


class BeliefTracking(object):
    look_ahead_time = 0.33
    look_ahead_points = 10
    filter_gain = 0.75

    def __init__(self, car, other_cars, tracker_type=PositionTracking, normalize=True):
        self.car = car
        self.tracker = tracker_type(car)
        self.other_cars = [] if other_cars is None else other_cars
        self.normalize = normalize
        # x_m, y_m, theta_m, v, k, k_t, k_a
        high = [  checkpoint_median_length,  half_track_width,  pi, car.specs.v_max,  car.specs.K_max,  max_curvature,  max_curvature ]
        low  = [ -checkpoint_median_length, -half_track_width, -pi,             0.0, -car.specs.K_max, -max_curvature, -max_curvature ]
        self.sensor_index = len(low)
        self.average_index = self.sensor_index + len(car.dist_sensors)
        for _ in range(2):
            for s in car.dist_sensors:
                high.append(s.specs[0])
                low.append(0.0)
        high, low = np.array(high, dtype=np.float32), np.array(low, dtype=np.float32)
        self._high = high
        if normalize:
            low  = low / high
            high = high / high
        self.observation_space = spaces.Box(low, high)
        self.belief = np.zeros(low.shape, dtype=low.dtype)
        self.dist = np.zeros(len(car.dist_sensors), dtype=low.dtype)
        self._reset_filters()

    def _read_sensors(self):
        for i in range(len(self.car.dist_sensors)):
            self.dist[i] = self.car.dist_sensors[i].read(self.other_cars)
        self.belief[self.sensor_index:self.average_index] = self.dlp.filter(self.dist)
        self.belief[self.average_index:] = self.dma.filter(self.dist)

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
    
    def _reset_filters(self):
        self.dlp = LowPassFilter(self.filter_gain, self.dist)
        self.dma = MovingAverage(0.33, 1.0/60.0)

    def reset(self, x_m, observation):
        pos = self.tracker.reset(x_m, observation)
        bel = self._augment_pos(pos)
        self._reset_filters()
        return bel

    def update(self, action, observation, dt):
        pos = self.tracker.update(action, observation, dt)
        return self._augment_pos(pos)
