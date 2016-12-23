# -*- coding: utf-8 -*-
"""
@author: Jean-Claude Manoli
"""

import math
import numpy as np
from numpy import cos, sin, pi
from utils import *
from gym.envs.classic_control import rendering

steer_actions =    [ 0.,  1., -1.,  0.,  1., -1.,  0.,  1., -1.]
throttle_actions = [ 0.,  0.,  0.,  1.,  1.,  1., -1., -1., -1.]

class CarSpecifications():
    carwidth = 0.12
    carlenght = 0.24
    steer_step = 0.1
    throttle_step = 0.1
    K_max = 4.5
    v_max = 2.5
    diag_angle = math.atan2(carwidth, carlenght)


class Car():
    
    breadcrumb = None

    def __init__(self, color=Color.black, specs=CarSpecifications(), trail_length=180, *args, **kwargs):
        self.specs = specs
        self.color = color
        self.trail_length = trail_length
        self.reset(*args, **kwargs)


    def reset(self, x=0.0, y=0.0, theta=0.0, steer=0.0, throttle=0.0, odometer=0.0, v=0.0):
        self.state = (x, y, theta, steer, throttle, odometer, v)
        if self.breadcrumb != None:
            self.breadcrumb.v.clear()


    def _dsdt(self, s, t, a, K_dot):
        '''
        Computes derivatives of state parameters.
        '''
        x, y, theta, v, K, d = s
        x_dot = v * cos(theta)
        y_dot = v * sin(theta)
        theta_dot = v * K
        return x_dot, y_dot, theta_dot, a, K_dot, v


    def _move(self, x, y, theta, v, K, d, a, K_dot, dt):
        s = x, y, theta, v, K, d
        I = rk4(self._dsdt, s, [0.0, dt], a, K_dot)
        x, y, theta, v, K, d = I[1]
        theta = canonical_angle(theta)
        return x, y, theta, v, K, d


    def reset_odometer(self, value):
        x=2


    def step(self, action, dt):
        '''
        Executes the specified action and computes the new state.
        '''

        # initial state
        x_, y_, theta_, steer_, throttle_, d_, v_ = self.state
        v_ = self.specs.v_max * throttle_
        K_ = self.specs.K_max * steer_

        # action
        ds = steer_actions[action] * self.specs.steer_step
        steer = max(-1.0, min(1.0, steer_ + ds))

        dp = throttle_actions[action] * self.specs.throttle_step
        throttle = self._safe_throttle_move(steer_ + ds, throttle_, dp)
        v = self.specs.v_max * throttle

        a = self.specs.v_max * (throttle - throttle_) / dt
        K_dot = self.specs.K_max * (steer - steer_) / dt

        # get new state
        x, y, theta, _, _, d = self._move(x_, y_, theta_, v_, K_, d_, a, K_dot, dt)

        self.state = (x, y, theta, steer, throttle, d, v)

        return self.state


    def car_min_distance(self, cars):
        dists = self.car_distances(cars)
        i = np.argmin([d for d, _, _ in dists])
        return dists[i]
        

    def car_distances(self, cars):
        distances = []
        for c in cars:
            if c == self:
                continue
            d, alpha = self.car_distance(c)
            distances.append((d, alpha, c))
        return distances
        

    def car_distance(self, car2):
        x1, y1, th1, _, _, _, _ = self.state
        x2, y2, th2, _, _, _, _ = car2
        d, alpha = self.distance(x2, y2)
        alpha2 = th2 - th1 + alpha 
        bd2 = _bumper_distance(car2, alpha2)
        return d - bd2, alpha1
        

    def distance(self, x, y):
        x1, y1, th1, _, _, _, _ = self.state
        dx = x - x1
        dy = y - y1
        dc = math.sqrt(dx ** 2 + dy ** 2)
        alpha = math.atan2(dy, dx) - th1
        bd = self._bumper_distance(alpha)
        d = dc - bd
        return d, alpha
        

    def _bumper_distance(self, alpha):
        l = self.specs.carlenght / 2.0
        w = self.specs.carwidth / 2.0
        alpha = abs(wrap(alpha, -pi / 2.0, pi / 2.0))
        if alpha <= self.specs.diag_angle:
            return l / cos(alpha)
        else:
            return w / sin(alpha)


    def _safe_throttle_move(self, steer, throttle, desired_change):
        '''
        Moves the throttle by the desired amout or according to the safe speed limit.
        '''
        safe = self._safe_throttle(steer)
        if throttle + desired_change > safe:
            return safe
        else:
            return max(0.0, min(1.0, throttle + desired_change))

    def _safe_throttle(self, steer):
        '''
        Gets the safe throttle value based on the specified steering.
        '''
        return min(1.0, 1.0 - abs(steer) / 2.0)


    def init_rendering(self, viewer):
        self.breadcrumb = rendering.PolyLine([], close=False)
        self.breadcrumb.set_color(*self.color)
        viewer.add_geom(self.breadcrumb)

        l = -self.specs.carlenght / 2.0
        r = self.specs.carlenght / 2.0
        t = self.specs.carwidth / 2.0
        b = -self.specs.carwidth / 2.0

        self.cartrans = rendering.Transform()

        car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        car.set_color(255, 64, 128)
        car.add_attr(self.cartrans)
        viewer.add_geom(car)

        d = 0.015
        sensor = rendering.FilledPolygon([(-d, -d), (-d, +d), (+d, +d), (+d, -d)])
        sensor.set_color(255, 0, 0)
        sensor.add_attr(self.cartrans)
        viewer.add_geom(sensor)

        steer = rendering.PolyLine([(-0.035, 0.0), (0.035, 0.0)], close=False)
        steer.set_linewidth(3)
        steer.set_color(0, 0, 255)
        self.steertrans = rendering.Transform()
        self.steertrans.set_translation(0.065, 0.0)
        steer.add_attr(self.steertrans)
        steer.add_attr(self.cartrans)
        viewer.add_geom(steer)

        carout = rendering.PolyLine([(l, b), (l, t), (r, t), (r, b)], close=True)
        carout.set_linewidth(3)
        carout.set_color(*self.color)
        carout.add_attr(self.cartrans)
        viewer.add_geom(carout)


    def render(self):
        x, y, theta, steer, _, _, _ = self.state
        self.cartrans.set_translation(x, y)
        self.cartrans.set_rotation(theta)
        self.steertrans.set_rotation(steer * pi / 2.0)
        self.breadcrumb.v.append((x, y))
        if len(self.breadcrumb.v) > self.trail_length:
            self.breadcrumb.v.pop(0)
