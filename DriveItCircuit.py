# -*- coding: utf-8 -*-
"""
Drive-It competition simulation environment map
@author: Jean-Claude Manoli
"""
import math
import numpy as np
from numpy import cos, sin, pi
from utils import right_angle

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
blue_width = 0.15

def cartesian_to_median(x: float, y: float, theta: float):
    '''
    Calculates the median coordinates of the specified position.
    '''
    # on central cross
    if abs(x) <= median_radius and abs(y) <= median_radius:

        if theta < - pi:
            x_m = -checkpoint_median_length + y + half_track_width
            y_m = -x
            tangent = - 3.0 * pi / 2.0
        else:
            x_m = x + half_track_width
            y_m = y
            tangent = 0.0

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
        tangent = -pi + np.arctan2(dx, dy)
        x_m = -loop_median_length - threshold_offset + (tangent + 3.0 * pi / 2.0) * median_radius
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
        if x_m < -loop_median_length - threshold_offset:
            tangent = -3.0 * pi / 2.0
            x = -y_m
            y = x_m + checkpoint_median_length - half_track_width
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
            tangent = (median_radius - x_m) / median_radius
            return tangent, -loop_curvature

    # after checkpoint
    else:
        # checkpoint straight line
        if x_m < -loop_median_length - half_track_width:
            return - 3.0 * pi / 2.0, 0.0
        # upper-left loop
        else:
            tangent =  (x_m - threshold_offset) / median_radius
            return tangent, loop_curvature


def median_position(x: float, y: float, current_mdist: float):
    '''
    Calculates the normalized longitudinal position along the track.

    Returns (x_m, y_m, lap, checkpoint) where:
    x_m: is the longitudinal position along the track,
    y_m: is the lateral distance from the track median,
    lap: is True if the car just passed the lap threshold
    checkpoint: is True if the car just passed the checkpoint threshold
    '''

    # on central cross
    if abs(x) <= median_radius and abs(y) <= median_radius:

        # lap straight line
        if current_mdist > - loop_median_length and current_mdist <= loop_median_length:
            lap = current_mdist < 0
            return x + median_radius, y, lap, False

        # checkpoint straight line
        else:
            checkpoint = current_mdist > 0
            return -checkpoint_median_length + y + median_radius, -x, False, checkpoint

    # lower-right loop
    if x > -median_radius and y < median_radius:
        dx = x - median_radius
        dy = -y - median_radius
        alpha = np.arctan2(dy, dx) + right_angle
        y_m = math.sqrt(dx ** 2 + dy ** 2) - median_radius
        return line_length + alpha * median_radius, y_m, False, False

    # upper-left loop
    else:
        dy = y - median_radius
        dx = -x - median_radius
        alpha = np.arctan2(dx, dy) + right_angle
        y_m = median_radius - math.sqrt(dx ** 2 + dy ** 2)
        return -loop_median_length + alpha * median_radius, y_m, False, False


def curve_error(theta: float, K: float, x_m: float):
    '''
    Calculates the heading and curvature offsets with the specified track median position.
    '''

    # before checkpoint
    if x_m >= 0:
        # lap straight line
        if x_m < line_length:
            tangent = 0.0
            curvature = 0.0

        # lower-right loop
        else:
            tangent = (line_length - x_m) / median_radius
            curvature = -loop_curvature

    # after checkpoint
    else:
        # checkpoint straight line
        if x_m < -loop_median_length:
            tangent = right_angle
            curvature = 0.0

        # upper-left loop
        else:
            tangent = x_m / median_radius
            curvature = loop_curvature

    theta_m = tangent - theta
    K_err = curvature - K

    return theta_m, K_err
