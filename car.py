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


steer_actions =    [ 0.,  1., -1.]
throttle_actions = [ 0.,  1., -1.]


class CarSpecifications():
    car_width = 0.12
    car_length = 0.24
    steer_step = 0.05
    throttle_step = 0.05
    K_max = 4.5
    max_accel = 10.0

    def __init__(self, v_max=2.5):
        self.v_max = v_max


class Car(RectangularPart):
    
    breadcrumb = None

    def __init__(self, color=Color.black, specs=CarSpecifications(), trail_length=180, *args, **kwargs):
        RectangularPart.__init__(self, specs.car_length, specs.car_width)
        self.specs = specs
        self.color = color
        self.trail_length = trail_length
        self.dist_sensors = []
        self.reset(*args, **kwargs)


    def add_dist_sensor(self, sensor, x, y, theta):
        self.dist_sensors.append(sensor)
        self.add_part(sensor, x, y, theta)


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
        steer_action, throttle_action = action

        ds = steer_actions[steer_action] * self.specs.steer_step
        steer = max(-1.0, min(1.0, steer_ + ds))
        K = self.specs.K_max * steer

        dp = throttle_actions[throttle_action] * self.specs.throttle_step
        throttle = Car._safe_throttle_move(steer, throttle_, dp)

        deltav = self.specs.v_max * throttle - v_
        dvmax = self.specs.max_accel * dt
        if abs(deltav) > dvmax:
            deltav = math.copysign(dvmax, deltav)
        v = v_ + deltav

        a = (v - v_) / dt
        K_dot = (K - K_) / dt

        # get new state
        x, y, theta, _, _, d = Car._move(x_, y_, theta_, v_, K_, d_, a, K_dot, dt)

        self.set_position(x, y, theta)
        self.state = (steer, throttle, d, v, K)

        return x, y, theta, steer, throttle, d, v, K


    def detect_collision(self, cars):
        for car in cars:
            if car != self:

                if car.is_collided(self.front_left) \
                or car.is_collided(self.front_right):
                    return car

                if self.is_collided(car.back_left) \
                or self.is_collided(car.back_right):
                    return car

        return None


    def get_velocity(self):
        v = self.state[3]
        th = self._position[2]
        return v, th


    def set_velocity(self, v):
        steer, throttle, d, _, K = self.state
        self.state = steer, throttle, d, v, K


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
        self.breadcrumb._color.vec4 = Color.set_alpha(self.color, 0.5);
        viewer.add_geom(self.breadcrumb)


    def get_geometry(self):

        l = -self.length / 2.0
        r = self.length / 2.0
        t = self.width / 2.0
        b = -self.width / 2.0

        body = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        body.set_color(1, 1, 1)

        d = 0.015
        sensor = rendering.FilledPolygon([(-d, -d), (-d, +d), (+d, +d), (+d, -d)])
        sensor.set_color(1, 0, 0)

        bumpers = rendering.PolyLine([(l, b), (l, t), (r, t), (r, b)], close=True)
        bumpers.set_linewidth(3)
        bumpers._color.vec4 = self.color

        return [body, sensor, bumpers]


    def render(self, viewer):
        
        self.steering_wheel.set_rotation(self.state[0] * right_angle)
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

