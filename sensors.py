# -*- coding: utf-8 -*-
"""
Distance sensor for the DriveIt Gym environment.
@author: Jean-Claude Manoli
"""

import math
import numpy as np
from numpy import cos, sin, pi
from utils import *
from part import *
from gym.envs.classic_control import rendering


class DistanceSensor(Part):

    def __init__(self, range_max, range_min, cone, precision):
        Part.__init__(self)
        self.specs = (range_max, range_min, cone, precision)
        self.np_random = np.random


    def long_range():
        return DistanceSensor(0.8, 0.03, 20 / 360. * pi, 0.01)

    def mid_range():
        return DistanceSensor(0.4, 0.01, 20 / 360. * pi, 0.002)

    def short_range():
        return DistanceSensor(0.2, 0.01, 20 / 360. * pi, 0.001)

    def wide():
        return DistanceSensor(1.0, 0.03, pi / 4, 0.01)


    def set_random(self, np_random):
        self.np_random = np_random


    def read(self, parts):

        range_max, range_min, cone, precision = self.specs
        dist = range_max
        for part_dist, alpha, part in self.part_distances(parts):
            if part == self.parent: continue
            if part_dist < dist:
                if abs(alpha) <= cone:
                    dist = part_dist
                else:
                    visa = part.visible_arc(alpha)
                    arc = math.atan2(visa, part_dist)
                    if abs(alpha) - arc <= cone:
                        dist = part_dist

        dist =self.np_random.normal(dist, precision)

        if dist < range_min:
            return (range_min - dist) * range_max
        else:
            return dist


    def get_geometry(self):
        d = 0.0075
        sensor = rendering.FilledPolygon([(-d, -d), (-d, +d), (+d, +d), (+d, -d)])
        sensor.set_color(1, 0, 0)

        range_max, range_min, arc, precision = self.specs
        ry = range_max * sin(arc)
        rx = range_max * cos(arc)
        cone = rendering.PolyLine([(rx, ry), (0., 0.), (rx, -ry)], close=False)
        cone.set_linewidth(1)
        cone._color.vec4 = (1, 0, 0, 0.3)

        return [sensor, cone]


class LidarSensor(Part):

    def __init__(self, range_max, distance_resolution=0.001, angular_resolution=0.01):
        Part.__init__(self)
        self.specs = (range_max, distance_resolution, angular_resolution)
        self.noisy = True
        self.np_random = np.random

    def set_noise(self, noisy, np_random):
        self.noisy = noisy
        self.np_random = np_random

    def read(self, parts):
        range_max, distance_resolution, angular_resolution = self.specs
        dist = []
        for part_dist, alpha, part in self.part_distances(parts):
            if part == self.parent: continue
            if part_dist <= range_max:
                if self.noisy:
                    part_dist =self.np_random.normal(part_dist, distance_resolution)
                    alpha =self.np_random.normal(alpha, angular_resolution)
                dist.append((part_dist, alpha))
        return dist

    def get_geometry(self):
        r = 0.025
        frame = rendering.make_circle(r, res=12, filled=True)
        frame.set_color(0.4, 0.4, 0.4)
        sensor = rendering.make_circle(r, res=12, filled=False)
        sensor.set_linewidth(2)
        sensor.set_color(1.0, 0.0, 0.0)

        return [frame, sensor]
