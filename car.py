# -*- coding: utf-8 -*-
"""
Car class for the DriveIt Gym environment.
@author: Jean-Claude Manoli
"""

import math
import numpy as np
from numpy import cos, sin, pi
from utils import *
from part import *
from gym.envs.classic_control import rendering


steer_actions =    [ 0.,  1., -1.,  0.,  1., -1.,  0.,  1., -1.]
throttle_actions = [ 0.,  0.,  0.,  1.,  1.,  1., -1., -1., -1.]


class CarSpecifications():
    car_width = 0.12
    car_lenght = 0.24
    diag_angle = math.atan2(car_width, car_lenght)
    steer_step = 0.1
    throttle_step = 0.1
    K_max = 4.5

    def __init__(self, v_max=2.5):
        self.v_max = v_max


class Car(Part):
    
    breadcrumb = None

    def __init__(self, color=Color.black, specs=CarSpecifications(), trail_length=180, *args, **kwargs):
        Part.__init__(self)
        self.specs = specs
        self.color = color
        self.trail_length = trail_length
        self.reset(*args, **kwargs)


    def reset(self, x=0.0, y=0.0, theta=0.0, steer=0.0, throttle=0.0, odometer=0.0, v=None):
        if v is None:
            v = self.specs.v_max * throttle
        K = self.specs.K_max * steer

        self.set_position(x, y, theta)
        self.state = (steer, throttle, odometer, v, K)
        if self.breadcrumb != None:
            self.breadcrumb.v.clear()

        return x, y, theta, steer, throttle, odometer, v, K


    def _dsdt(s, t, a, K_dot):
        '''
        Computes derivatives of state parameters.
        '''
        x, y, theta, v, K, d = s
        x_dot = v * cos(theta)
        y_dot = v * sin(theta)
        theta_dot = v * K
        return x_dot, y_dot, theta_dot, a, K_dot, v


    def _move(x, y, theta, v, K, d, a, K_dot, dt):
        s = x, y, theta, v, K, d
        I = rk4(Car._dsdt, s, [0.0, dt], a, K_dot)
        x, y, theta, v, K, d = I[1]
        theta = canonical_angle(theta)
        return x, y, theta, v, K, d


    def reset_odometer(self, value):
        steer, throttle, d, v, K = self.state
        self.state = (steer, throttle, value, v, K)


    def step(self, action, dt):
        '''
        Executes the specified action and computes the new state.
        '''

        # initial state
        x_, y_, theta_ = self.get_position()
        steer_, throttle_, d_, v_, K_ = self.state

        # action
        ds = steer_actions[action] * self.specs.steer_step
        steer = max(-1.0, min(1.0, steer_ + ds))
        K = self.specs.K_max * steer

        dp = throttle_actions[action] * self.specs.throttle_step
        throttle = Car._safe_throttle_move(steer, throttle_, dp)
        v = self.specs.v_max * throttle

        a = (v - v_) / dt
        K_dot = (K - K_) / dt

        # get new state
        x, y, theta, _, _, d = Car._move(x_, y_, theta_, v_, K_, d_, a, K_dot, dt)

        self.set_position(x, y, theta)
        self.state = (steer, throttle, d, v, K)

        return x, y, theta, steer, throttle, d, v, K


    def closest_car(self, cars):
        '''
        Returns the distance and angle to the closest car.
        '''
        if len(cars) > 0:
            dists = self.car_distances(cars)
            i = np.argmin([d for d, _, _ in dists])
            return dists[i]
        else:
            return None
        

    def car_distances(self, cars):
        '''
        Returns the distances and angles to the specified list of cars.
        '''
        distances = []
        for c in cars:
            if c == self:
                continue
            d, alpha = self.car_distance(c)
            distances.append((d, alpha, c))
        return distances
        

    def car_distance(self, car):
        '''
        Calculates the distance and relative angle to the specified car.
        '''
        x1, y1, th1 = self.get_position()
        x2, y2, th2 = car.get_position()
        d, alpha = self.distance(x2, y2)
        alpha2 = th2 - th1 + alpha 
        bd2 = car._bumper_distance(alpha2)
        return d - bd2, alpha
        

    def distance(self, x, y):
        '''
        Calculates the distance and relative angle to the specified location.
        '''
        x1, y1, th1 = self.get_position()
        dx = x - x1
        dy = y - y1
        dc = math.sqrt(dx ** 2 + dy ** 2)
        alpha = math.atan2(dy, dx) - th1
        bd = self._bumper_distance(alpha)
        d = dc - bd
        return d, alpha
        

    def _bumper_distance(self, alpha):
        '''
        Calculates the car's center to bumper distance for the specified angle.
        '''
        l = self.specs.car_lenght / 2.0
        w = self.specs.car_width / 2.0
        alpha = abs(wrap(alpha, -pi / 2.0, pi / 2.0))
        if alpha <= self.specs.diag_angle:
            return l / cos(alpha)
        else:
            return w / sin(alpha)


    def _safe_throttle_move(steer, throttle, desired_change):
        '''
        Moves the throttle by the desired amout or according to the safe speed limit.
        '''
        safe = Car._safe_throttle(steer)
        if throttle + desired_change > safe:
            return safe
        else:
            return max(0.0, min(1.0, throttle + desired_change))

    def _safe_throttle(steer):
        '''
        Gets the safe throttle value based on the specified steering.
        '''
        return min(1.0, 1.0 - abs(steer) / 2.0)


    def init_rendering(self, viewer):

        self.steering_wheel = SteeringWheel()
        self.add_part(self.steering_wheel, 0.065, 0.0, 0.0)
        self.breadcrumb = rendering.PolyLine([], close=False)
        self.breadcrumb.set_color(*self.color)
        viewer.add_geom(self.breadcrumb)


    def get_geometry(self):

        l = -self.specs.car_lenght / 2.0
        r = self.specs.car_lenght / 2.0
        t = self.specs.car_width / 2.0
        b = -self.specs.car_width / 2.0

        body = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        body.set_color(255, 64, 128)

        d = 0.015
        sensor = rendering.FilledPolygon([(-d, -d), (-d, +d), (+d, +d), (+d, -d)])
        sensor.set_color(255, 0, 0)

        bumpers = rendering.PolyLine([(l, b), (l, t), (r, t), (r, b)], close=True)
        bumpers.set_linewidth(3)
        bumpers.set_color(*self.color)

        return [body, sensor, bumpers]


    def render(self, viewer):
        
        self.steering_wheel.set_rotation(self.state[0] * pi / 2.0)
        x, y, theta = self.get_position()
        self.breadcrumb.v.append((x, y))
        if len(self.breadcrumb.v) > self.trail_length:
            self.breadcrumb.v.pop(0)

        Part.render(self, viewer)



class SteeringWheel(Part):
    def get_geometry(self):
        steer = rendering.PolyLine([(-0.035, 0.0), (0.035, 0.0)], close=False)
        steer.set_linewidth(3)
        steer.set_color(0, 0, 255)
        return [steer]

