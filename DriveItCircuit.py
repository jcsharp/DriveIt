# -*- coding: utf-8 -*-
"""
Drive-It competition simulation environment map
@author: Jean-Claude Manoli
"""
import math
import numpy as np
from numpy import cos, sin, pi
from utils import right_angle, canonical_angle

# track metrics
median_radius = 0.375
loop_curvature = 1.0 / median_radius
line_length = 2.0 * median_radius
loop_median_length = 3.0 / 2.0 * pi * median_radius
checkpoint_median_length = line_length + loop_median_length
lap_median_length = 2.0 * checkpoint_median_length
half_track_width = 0.225
blue_width = 0.15

def cartesian_to_median(x: float, y: float, theta: float, checkpoint: bool):
    '''
    Calculates the median coordinates of the specified position.
    '''
    # on central cross
    if abs(x) <= median_radius and abs(y) <= median_radius:

        if checkpoint:
            x_m = -checkpoint_median_length + y + median_radius
            y_m = -x
            tangent = right_angle
        else:
            x_m = x + median_radius
            y_m = y
            tangent = 0.0

    # lower-right loop
    elif x > -median_radius and y < median_radius:
        dx = x - median_radius
        dy = -y - median_radius
        tangent = -np.arctan2(dy, dx) - right_angle
        x_m = line_length - tangent * median_radius
        y_m = math.sqrt(dx ** 2 + dy ** 2) - median_radius

    # upper-left loop
    else:
        dy = y - median_radius
        dx = -x - median_radius
        tangent = -np.arctan2(dx, dy) - right_angle
        x_m = -loop_median_length - tangent * median_radius
        y_m = median_radius - math.sqrt(dx ** 2 + dy ** 2)

    theta_m = canonical_angle(tangent - theta)

    return x_m, y_m, theta_m


def median_to_cartesian(x_m: float, y_m: float, theta_m: float):
    '''
    Calculates the cartesian coordinates of a specific position relative to the track median.
    '''

    # before checkpoint
    if x_m >= 0:
        alpha = (x_m - line_length) / median_radius
        theta = canonical_angle(theta_m - alpha)
        # lap straight line
        if x_m < line_length:
            return x_m - median_radius, y_m, theta
        # lower-right loop
        else:
            x = (median_radius + y_m) * sin(alpha) + median_radius
            y = (median_radius + y_m) * cos(alpha) - median_radius
            return x, y, theta

    # after checkpoint
    else:
        alpha = -x_m / median_radius
        theta = canonical_angle(theta_m - alpha)
        # checkpoint straight line
        if x_m < -loop_median_length:
            return -y_m, x_m + checkpoint_median_length - median_radius, theta
        # upper-left loop
        else:
            x = (y_m - median_radius) * sin(alpha) - median_radius
            y = (y_m - median_radius) * cos(alpha) + median_radius
            return x, y, theta


def median_properties(x_m: float):
    '''
    Calculates the tangent and curvature of a specific position on the track median.
    '''
    # before checkpoint
    if x_m >= 0:
        # lap straight line
        if x_m < line_length:
            return 0.0, 0.0
        # lower-right loop
        else:
            alpha = (x_m - line_length) / median_radius
            return canonical_angle(-alpha), -loop_curvature

    # after checkpoint
    else:
        # checkpoint straight line
        if x_m < -loop_median_length:
            return right_angle, 0.0
        # upper-left loop
        else:
            alpha = -x_m / median_radius
            return canonical_angle(-alpha), loop_curvature


def lateral_error(x: float, y: float, x_m: float):
    '''
    Calculates the lateral distance between the car center and the track median.
    '''

    # before checkpoint
    if x_m >= 0:
        # lap straight line
        if x_m < line_length:
            y_m = y

        # lower-right loop
        else:
            dx = x - median_radius
            dy = y + median_radius
            y_m = math.sqrt(dx ** 2 + dy ** 2) - median_radius

    # after checkpoint
    else:
        # checkpoint straight line
        if x_m < -loop_median_length:
            y_m = -x

        # upper-left loop
        else:
            dx = x + median_radius
            dy = y - median_radius
            y_m = median_radius - math.sqrt(dx ** 2 + dy ** 2)

    return y_m

def median_distance(x: float, y: float, current_mdist: float):
    '''
    Calculates the normalized longitudinal position along the track.

    Returns (x_m, lap, checkpoint) where:
    x_m: is the normalized longitudinal position along the track,
    lap: is True if the car just passed the lap threshold
    checkpoint: is True if the car just passed the checkpoint threshold
    '''

    # on central cross
    if abs(x) <= median_radius and abs(y) <= median_radius:

        # lap straight line
        if current_mdist > - loop_median_length and current_mdist <= loop_median_length:
            lap = current_mdist < 0
            return x + median_radius, lap, False

        # checkpoint straight line
        else:
            checkpoint = current_mdist > 0
            return -checkpoint_median_length + y + median_radius, False, checkpoint

    # lower-right loop
    if x > -median_radius and y < median_radius:
        dx = x - median_radius
        dy = -y - median_radius
        alpha = np.arctan2(dy, dx) + right_angle
        return line_length + alpha * median_radius, False, False

    # upper-left loop
    else:
        dy = y - median_radius
        dx = -x - median_radius
        alpha = np.arctan2(dx, dy) + right_angle
        return -loop_median_length + alpha * median_radius, False, False
