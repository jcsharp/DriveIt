# -*- coding: utf-8 -*-
"""
Car class for the DriveIt Gym environment.
@author: Jean-Claude Manoli
"""

import math
import numpy as np
from numpy import cos, sin, pi, sqrt, clip, sign #pylint: disable=E0611
from utils import *
from part import *
from sensors import *
from gym.envs.classic_control import rendering


max_steer_bias = 0.05
max_throttle_bias = 0.05


class CarSpecifications():
    car_width = 0.12
    car_length = 0.24
    max_steer_speed = 5.0
    K_max = 4.5
    max_accel = 10.0
    lateral_offset = lateral_offset_default = 0.0

    def __init__(self, v_max=2.5):
        self.v_max = self.v_max_default = v_max

    def set_lateral_offset(self, value):
        self.lateral_offset = self.lateral_offset_default = value

    def limit_steer_move(self, desired, current, dt):
        ss = (desired - current) / dt
        if abs(ss) > self.max_steer_speed:
            return current + self.max_steer_speed * dt * sign(ss)
        else:
            return desired

    def limit_throttle_move(self, desired, current, dt):
        a = (desired - current) * self.v_max / dt
        if abs(a) > self.max_accel:
            return current + self.max_accel / self.v_max * dt * sign(a)
        else:
            return desired

    def safe_turn_speed(self, steer, margin=1.0):
        '''
        Gets the safe velocity based on the specified steering position.
        '''
        if steer == 0.0: return self.v_max
        safe = sqrt(self.max_accel / self.K_max / abs(steer)) * margin
        return min(self.v_max, safe)

    def curvature_steer(self, K):
        '''
        Get the steering position for the specified curvature.
        '''
        return clip(K / self.K_max, -1.0, 1.0)

        

class Car(RectangularPart):

    def Simple(color=Color.black, v_max=1.0): #pylint: disable=E0213
        car = Car(color, CarSpecifications(v_max))
        car.add_dist_sensor(DistanceSensor.wide(), 0.06, 0.0, 0)
        return car


    def HighPerf(color=Color.black, v_max=2.5): #pylint: disable=E0213
        car = Car(color, CarSpecifications(v_max))
        car.add_dist_sensor(DistanceSensor.long_range(), 0.115, 0.0, 0.0)
        car.add_dist_sensor(DistanceSensor.mid_range(), 0.115, 0.02, pi / 6.0)
        car.add_dist_sensor(DistanceSensor.mid_range(), 0.115, -0.02, -pi / 6.0)
        car.add_dist_sensor(DistanceSensor.mid_range(), 0.0, 0.055, pi / 4.0)
        car.add_dist_sensor(DistanceSensor.mid_range(), 0.0, -0.055, -pi / 4.0)
        car.add_part(ColorSensor(), 0.0, 0.0, 0.0)
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
        self.position = 0.0, 0.0, 0.0

    def set_noise(self, noisy, np_random):
        self.noisy = noisy
        self.np_random = np_random
        for s in self.dist_sensors:
            s.set_random(noisy, np_random)


    def add_dist_sensor(self, sensor, x, y, theta):
        self.dist_sensors.append(sensor)
        self.add_part(sensor, x, y, theta)


    def reset(self, x=0.0, y=0.0, theta=0.0, steer=0.0, throttle=0.0, v=None):
        if v is None:
            v = self.specs.v_max * throttle
        K = self.specs.K_max * steer

        self.position = x, y, theta
        self.set_position(x, y, theta)
        self.state = steer, throttle, v, K
        self.bias = 0.0, 0.0
        if self.breadcrumb != None:
            self.breadcrumb.v.clear()


    def _dsdt(self, s, t, a, K_dot):
        '''
        Computes derivatives of state parameters.
        '''
        _, _, theta, v, K = s
        x_dot = v * cos(theta)
        y_dot = v * sin(theta)
        theta_dot = v * K
        return x_dot, y_dot, theta_dot, a, K_dot


    def _move(self, x, y, theta, v, K, a, K_dot, dt):
        s = x, y, theta, v, K
        I = rk4(self._dsdt, s, [0.0, dt], a, K_dot)
        x, y, theta, v, K = I[1]
        return x, y, theta, v, K


    def step(self, action, dt):
        '''
        Executes the specified action and computes the new state.
        '''

        # initial state
        x_, y_, theta_ = self.position
        steer_, throttle_, v_, K_ = self.state

        # action
        steer, throttle = action
        steer = self.specs.limit_steer_move(steer, steer_, dt)
        throttle = self.specs.limit_throttle_move(throttle, throttle_, dt)
        throttle, throttle_override = self._safe_throttle_move(steer, throttle)

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
            if steer != steer_:
                steer_bias = self.np_random.uniform(-max_steer_bias, max_steer_bias)
            if throttle != throttle_:
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
        x, y, theta, _, _ = self._move(x_, y_, theta_, v_, K_, a, K_dot, dt)

        self.position = x, y, theta
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


    def _safe_throttle_move(self, steer, throttle):
        '''
        Moves the throttle to the desired position or according to the safe speed limit.
        '''
        safe = self.safe_throttle(steer)
        if throttle < safe:
            safe = clip(throttle, 0.0, 1.0)
        return safe, throttle - safe


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

        bumpers = rendering.PolyLine([(l, b), (l, t), (r, t), (r, b)], close=True)
        bumpers.set_linewidth(3)
        bumpers._color.vec4 = self.color

        return [body, bumpers]


    def render(self, viewer):
        
        self.steering_wheel.set_rotation(self.state[0] * right_angle)
        x, y, theta = self.position
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


class ColorSensor(Part):
    def get_geometry(self):
        d = 0.015
        sensor = rendering.FilledPolygon([(-d, -d), (-d, +d), (+d, +d), (+d, -d)])
        sensor.set_color(1, 0, 0)
        return [sensor]
