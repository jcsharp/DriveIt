# -*- coding: utf-8 -*-
"""
Car class for the DriveIt Gym environment.
@author: Jean-Claude Manoli
"""

import math
import numpy as np
from numpy import cos, sin, pi, sqrt, clip
from utils import *
from part import *
from sensors import *
from gym.envs.classic_control import rendering


steer_actions =    [ 0.,  1., -1.,  0.,  1., -1.,  0.,  1., -1.]
throttle_actions = [ 0.,  0.,  0.,  1.,  1.,  1., -1., -1., -1.]
max_steer_bias = 0.1
max_throttle_bias = 0.1


class CarSpecifications():
    car_width = 0.12
    car_length = 0.24
    steer_step = 0.1
    throttle_step = 0.1
    K_max = 4.5
    max_accel = 10.0

    def __init__(self, v_max=2.5):
        self.v_max = v_max

    def safe_turn_speed(self, steer, margin=1.0):
        '''
        Gets the safe velocity based on the specified steering position.
        '''
        if steer == 0.0: return self.v_max
        safe = sqrt(self.max_accel / self.K_max / abs(steer)) * margin
        return min(self.v_max, safe)
        

class Car(RectangularPart):

    def Simple(color=Color.black, v_max=1.0):
        car = Car(color, CarSpecifications(v_max))
        car.add_dist_sensor(DistanceSensor(1., 0.03, pi / 4, 0.01), 0.06, 0., 0)
        return car


    def HighPerf(color=Color.black, v_max=2.5):
        car = Car(color, CarSpecifications(v_max))
        car.add_dist_sensor(DistanceSensor.long_range(), 0.115, 0., 0.)
        car.add_dist_sensor(DistanceSensor.short_range(), 0.115, 0.02, pi / 6.)
        car.add_dist_sensor(DistanceSensor.short_range(), 0.115, -0.02, -pi / 6.)
        car.add_dist_sensor(DistanceSensor.short_range(), 0., 0.055, pi / 4.)
        car.add_dist_sensor(DistanceSensor.short_range(), 0., -0.055, -pi / 4.)
        return car
    
    
    breadcrumb = None

    def __init__(self, color=Color.black, specs=CarSpecifications(), trail_length=180, *args, **kwargs):
        RectangularPart.__init__(self, specs.car_length, specs.car_width)
        self.specs = specs
        self.color = color
        self.trail_length = trail_length
        self.dist_sensors = []
        self.noisy = False
        self.np_random = None
        self.reset(*args, **kwargs)


    def set_noise(self, noisy, np_random):
        self.noisy = noisy
        self.np_random = np_random
        for s in self.dist_sensors:
            s.set_random(np_random)


    def add_dist_sensor(self, sensor, x, y, theta):
        self.dist_sensors.append(sensor)
        self.add_part(sensor, x, y, theta)


    def reset(self, x=0.0, y=0.0, theta=0.0, steer=0.0, throttle=0.0, v=None):
        if v is None:
            v = self.specs.v_max * throttle
        K = self.specs.K_max * steer

        self.set_position(x, y, theta)
        self.state = steer, throttle, v, K
        self.bias = 0.0, 0.0
        if self.breadcrumb != None:
            self.breadcrumb.v.clear()

        return x, y, theta, steer, throttle, v, K


    def _dsdt(s, t, a, K_dot):
        '''
        Computes derivatives of state parameters.
        '''
        _, _, theta, v, K = s
        x_dot = v * cos(theta)
        y_dot = v * sin(theta)
        theta_dot = v * K
        return x_dot, y_dot, theta_dot, a, K_dot


    def _move(x, y, theta, v, K, a, K_dot, dt):
        s = x, y, theta, v, K
        I = rk4(Car._dsdt, s, [0.0, dt], a, K_dot)
        x, y, theta, v, K = I[1]
        return x, y, theta, v, K


    def step(self, action, dt):
        '''
        Executes the specified action and computes the new state.
        '''

        # initial state
        x_, y_, theta_ = self.get_position()
        steer_, throttle_, v_, K_ = self.state

        # action
        ds = steer_actions[action] * self.specs.steer_step
        steer = clip(steer_ + ds, -1.0, 1.0)
        dp = throttle_actions[action] * self.specs.throttle_step
        throttle, throttle_override = self._safe_throttle_move(steer, throttle_, dp)

        # desired state
        K_hat = self.specs.K_max * steer
        deltav = self.specs.v_max * throttle - v_
        dvmax = self.specs.max_accel * dt
        if abs(deltav) > dvmax:
            deltav = math.copysign(dvmax, deltav)
        v_hat = v_ + deltav

        # add mechanical noise
        if self.noisy:
            steer_bias, throttle_bias = self.bias
            if ds != 0.0:
                steer_bias = self.np_random.uniform(-max_steer_bias, max_steer_bias)
            if dp != 0.0:
                throttle_bias = self.np_random.uniform(-max_throttle_bias, max_throttle_bias)
            K = K_hat + self.specs.K_max * steer_bias
            v = v_hat * (1 + throttle_bias)
            self.bias = steer_bias, throttle_bias
        else:
            K = K_hat
            v = v_hat
            
        # get new state
        a = (v - v_) / dt
        K_dot = (K - K_) / dt
        x, y, theta, _, _ = Car._move(x_, y_, theta_, v_, K_, a, K_dot, dt)

        self.set_position(x, y, theta)
        self.state = steer, throttle, v, K

        return x, y, theta, steer, throttle, v, K_hat, throttle_override


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


    def _safe_throttle_move(self, steer, throttle, desired_change):
        '''
        Moves the throttle by the desired amout or according to the safe speed limit.
        '''
        desired_throttle = throttle + desired_change
        safe = self.safe_throttle(steer)
        if desired_throttle < safe:
            safe = clip(desired_throttle, 0.0, 1.0)
        return safe, desired_throttle - safe


    def safe_throttle(self, steer):
        '''
        Gets the safe throttle value based on the specified steering.
        '''
        return self.specs.safe_turn_speed(steer) / self.specs.v_max


    def init_rendering(self, viewer):

        self.steering_wheel = SteeringWheel()
        self.add_part(self.steering_wheel, 0.065, 0.0, 0.0)
        self.breadcrumb = rendering.PolyLine([], close=False)
        self.breadcrumb._color.vec4 = Color.set_alpha(self.color, 0.5)
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

