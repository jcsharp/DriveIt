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


    def long_range():
        return DistanceSensor(1., 0.03, 22.5 / 360. * pi, 0.01)

    def short_range():
        return DistanceSensor(0.25, 0.01, 22.5 / 360. * pi, 0.001)


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

        dist = np.random.normal(dist, precision)

        if dist < range_min:
            return (range_min - dist) * range_max
        else:
            return dist


    def get_geometry(self):
        d = 0.0075
        sensor = rendering.FilledPolygon([(-d, -d), (-d, +d), (+d, +d), (+d, -d)])
        sensor.set_color(1, 0, 0)

        range_max, range_min, cone, precision = self.specs
        ry = range_max * sin(cone) / 2.
        rx = range_max * cos(cone)
        cone = rendering.PolyLine([(rx, ry), (0., 0.), (rx, -ry)], close=False)
        cone.set_linewidth(1)
        cone._color.vec4 = (1, 0, 0, 0.3)

        return [sensor, cone]
