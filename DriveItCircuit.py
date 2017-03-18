# -*- coding: utf-8 -*-
"""
Drive-It competition simulation environment map
@author: Jean-Claude Manoli
"""
import math
import numpy as np
from numpy import cos, sin, pi
from utils import right_angle, three_quarter_turn

# track metrics
median_radius = 0.375
loop_curvature = 1.0 / median_radius
line_length = 2.0 * median_radius
loop_median_length = 3.0 / 2.0 * pi * median_radius
checkpoint_median_length = line_length + loop_median_length
lap_median_length = 2.0 * checkpoint_median_length
half_track_width = 0.225
threshold_offset = median_radius - half_track_width
threshold_to_curve = median_radius + half_track_width
checkpoint_to_lap = checkpoint_median_length - half_track_width
loop_to_threshold = loop_median_length + threshold_offset
blue_width = 0.15

def cartesian_to_median(x: float, y: float, theta: float):
    '''
    Calculates the median coordinates of the specified position.
    '''
    # on central cross
    if abs(x) <= median_radius and abs(y) <= median_radius:

        # lap straight line
        if theta > - pi:
            x_m = x + half_track_width
            y_m = y
            tangent = 0.0
        # checkpoint straight line
        else:
            x_m = y - checkpoint_to_lap
            y_m = -x
            tangent = - three_quarter_turn

    # lower-right loop
    elif x > -median_radius and y < median_radius:
        dx = x - median_radius
        dy = -y - median_radius
        tangent = -np.arctan2(dy, dx) - right_angle
        x_m = threshold_to_curve - tangent * median_radius
        y_m = math.sqrt(dx ** 2 + dy ** 2) - median_radius

    # upper-left loop
    else:
        dy = y - median_radius
        dx = -x - median_radius
        tangent = np.arctan2(dx, dy) - pi
        x_m = (tangent + three_quarter_turn) * median_radius - loop_to_threshold
        y_m = median_radius - math.sqrt(dx ** 2 + dy ** 2)

    theta_m = tangent - theta

    return x_m, y_m, theta_m


def median_to_cartesian(x_m: float, y_m: float, theta_m: float):
    '''
    Calculates the cartesian coordinates of a specific position relative to the track median.
    '''
    # before checkpoint
    if x_m >= -threshold_offset:
        # lap straight line
        if x_m < threshold_to_curve:
            tangent = 0.0
            x = x_m - half_track_width
            y = y_m
        # lower-right loop
        else:
            tangent = (half_track_width - x_m) / median_radius + 1
            x = (median_radius + y_m) * sin(-tangent) + median_radius
            y = (median_radius + y_m) * cos(-tangent) - median_radius

    # after checkpoint
    else:
        # checkpoint straight line
        if x_m < -loop_to_threshold:
            tangent = -three_quarter_turn
            x = -y_m
            y = x_m + checkpoint_to_lap
        # upper-left loop
        else:
            tangent = (x_m + threshold_offset) / median_radius
            x = (y_m - median_radius) * sin(-tangent) - median_radius
            y = (y_m - median_radius) * cos(-tangent) + median_radius

    theta = tangent - theta_m
    return x, y, theta


def median_properties(x_m: float):
    '''
    Calculates the tangent and curvature of a specific position on the track median.
    '''
    # before checkpoint
    if x_m >= -threshold_offset:
        # lap straight line
        if x_m < median_radius:
            return 0.0, 0.0
        # lower-right loop
        else:
            tangent = (threshold_to_curve - x_m) / median_radius
            return tangent, -loop_curvature

    # after checkpoint
    else:
        # checkpoint straight line
        if x_m < -loop_to_threshold:
            return - three_quarter_turn
        # upper-left loop
        else:
            tangent =  (x_m + threshold_offset) / median_radius
            return tangent, loop_curvature
