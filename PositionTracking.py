# -*- coding: utf-8 -*-
"""
Position tracking for the DriveIt Gym environment.
@author: Jean-Claude Manoli
"""
import numpy as np
from gym import spaces
from car import Car
from DriveItCircuit import * #pylint: disable=W0401,W0614
#pylint: disable=C0301


class PositionTrackingBase(object):

    def __init__(self, car):
        self.car = car
        high = np.array([  checkpoint_median_length,  half_track_width,  pi, car.specs.v_max,  car.specs.K_max ], dtype=np.float32)
        low  = np.array([ -checkpoint_median_length, -half_track_width, -pi,             0.0, -car.specs.K_max ], dtype=np.float32)
        self.observation_space = spaces.Box(low, high)
        self.observation = ()
        self.position = ()

    def reset(self, observation): raise NotImplementedError

    def update(self, action, observation, dt): raise NotImplementedError


class TruePosition(PositionTrackingBase):

    def _get_state(self):
        x, y, theta = self.car.position
        xm, ym, thm = cartesian_to_median(x, y, theta)
        _, _, v, K = self.car.state
        self.position = x, y, xm < 0.0
        return xm, ym, thm, v, K

    def reset(self, observation): #pylint: disable=W0613
        return self._get_state()

    def update(self, action, observation, dt): #pylint: disable=W0613
        return self._get_state()


class PositionTracking(PositionTrackingBase):

    def __init__(self, car):
        super().__init__(car)

    def reset(self, observation):
        x, y, _ = self.car.position
        blue, theta, v, K, checkpoint = observation #pylint: disable=W0612
        xm, ym, thm = cartesian_to_median(x, y, theta)
        self.observation = observation
        self.position = x, y, checkpoint
        return xm, ym, thm, v, K

    def update(self, action, observation, dt): #pylint: disable=W0613
        x_, y_, checkpoint_ = self.position
        _, theta, v, K, checkpoint = observation
        _, theta_, v_, K_, _ = self.observation
        self.observation = observation

        a = (v - v_) / dt
        K_dot = (K - K_) / dt
        x, y, _, _, _ = self.car._move(x_, y_, theta_, v_, K_, a, K_dot, dt)

        x_m, y_m, theta_m = cartesian_to_median(x, y, theta)

        # def print_change(x, y, xa, ya):
        #     def print_adjustment(name, value, new_value, real_value):
        #         change = new_value - value
        #         desired = real_value - value
        #         error = real_value - new_value
        #         print('%s adjusted by %f (ideal %f, err %f)' % (name, change, desired, error))
        #     if x != xa:
        #         print_adjustment('x', x, xa, self.car.position[0])
        #     if y != ya:
        #         print_adjustment('y', y, ya, self.car.position[1])        

        pos_adjusted, xa, ya = adjust_position(checkpoint != checkpoint_, checkpoint, x_m, x, y)
        if pos_adjusted:
            # print_change(x, y, xa, ya)
            x, y = xa, ya
            x_m, y_m, theta_m = cartesian_to_median(x, y, theta)

        self.position = x, y, checkpoint

        return x_m, y_m, theta_m, v, K
