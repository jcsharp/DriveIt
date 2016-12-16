# -*- coding: utf-8 -*-
"""
Drive-It competition simulation environment
@author: Jean-Claude Manoli
"""

import math
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np
from numpy import cos, sin, pi
from os import path

steer_step = 0.1
throttle_step = 0.1
steer_actions =    [ 0.,  1., -1.,  0.,  1., -1.,  0.,  1., -1.]
throttle_actions = [ 0.,  0.,  0.,  1.,  1.,  1., -1., -1., -1.]
blue_threshold = 0.75
fps = 60.0
dt = 1.0 / fps
K_max = 4.5
v_max = 2.5

# track metrics
median_radius = 0.375
line_length = 2.0 * median_radius
loop_median_length = 3.0 / 2.0 * pi * median_radius
checkpoint_median_length = line_length + loop_median_length
lap_median_length = 2.0 * checkpoint_median_length
half_track_width = 0.225
blue_width = 0.15
wrong_way_min = 0.275
wrong_way_max = median_radius

class DriveItEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': fps
    }


    def __init__(self, time_limit=10, throttle_limit=1.0, gamma=0.98, \
                 show_belief_state=False, trail_length=2.4):
        
        self.time_limit = time_limit
        self.throttle_limit = throttle_limit
        self.show_belief_state = show_belief_state
        self.trail_length = trail_length

        # corresponds to the maximum discounted reward over a median lap
        max_reward = throttle_limit * v_max * dt / (1 - gamma)
        self.out_reward = -max_reward
        
        self.viewer = None

        high = np.array([  1.0,  1.0,  1.0,  1.0, 1.0 ])
        low  = np.array([ -1.0, -1.0, -1.0, -1.0, 0.0 ])

        self.action_space = spaces.Discrete(len(steer_actions))
        self.observation_space = spaces.Box(low, high)

        fname = path.join(path.dirname(__file__), "track.png")
        self.track = rendering.Image(fname, 2., 2.)

        self._seed()
        self.time = -1.0
        self.reset()


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _reset(self, random_position=True):
        '''
        Resets the simulation.

        By default, the car is placed at a random position along the race track, 
        which improves learning.

        If random_position is set to False, the car is placed at the normal 
        starting position.
        '''

        if self.viewer:
            self.breadcrumb.v.clear()

        if random_position:
            # random position along the track median
            x_m = np.random.uniform(-checkpoint_median_length, checkpoint_median_length)
            y_m = np.random.uniform(-0.03, 0.03)
            x, y = self._median_to_cartesian(x_m, y_m)
            theta, steer = self._median_properties(x_m)

            # add some noise
            theta += np.random.uniform(-pi / 36.0, pi / 36.0)
            steer += np.random.randint(-1, 1) * steer_step
            throttle = int(self._safe_throttle(steer) * np.random.uniform() / throttle_step) * throttle_step

        else:
            # the default startup position, on the lap threshold
            x, y, theta, steer = -median_radius, 0.0, 0.0, 0.0
            x_m, y_m, blue_left, blue_right = 0.0, 0.0, 0.0, 0.0
            throttle = 0.0

        
        self.time = 0.0
        self.state = (x, y, theta, steer, throttle, x_m, y_m, x_m)
        self.belief = (x_m, y_m, theta, throttle * v_max, 0.0)
        if self.show_belief_state:
            observation = np.array((x_m / checkpoint_median_length, y_m / half_track_width, theta / pi, steer, throttle))
        else:
            observation = np.array((x_m / checkpoint_median_length, 0.0, theta / pi, steer, throttle))

        return observation


    def _dsdt(self, s, t):
        '''
        Computes derivatives of state parameters.
        '''
        x, y, theta, x_m, y_m, v, K, x_m_hat, y_m_hat, theta_hat, v_hat, dv, dK = s

        # position
        dx = v * cos(theta)
        dy = v * sin(theta)
        median_heading, _ = self._median_properties(x_m)
        err_heading = canonical_angle(theta - median_heading)
        dxm = v * cos(err_heading)
        dym = v * sin(err_heading)
        dtheta = v * K

        # belief
        median_heading, _ = self._median_properties(x_m_hat)
        err_heading = canonical_angle(theta_hat - median_heading)
        dxmh = v * cos(err_heading)
        dymh = v * sin(err_heading)
        dthetah = v_hat * K

        return dx, dy, dtheta, dxm, dym, dv, dK, dxmh, dymh, dthetah, dv, 0.0, 0.0


    def _step(self, action):
        '''
        Executes the specified action, computes a new state, observation, belief and reward.
        '''
        self.time += dt

        # initial state
        x, y, theta, steer_, throttle_, x_m_, y_m_, d = self.state
        x_m_hat_, y_m_hat_, theta_hat_, v_hat_, bias = self.belief
        v = v_max * throttle_
        K = K_max * steer_

        # action
        ds = steer_actions[action] * steer_step
        steer = max(min(steer_ + ds, 1.0), -1.0)

        dp = throttle_actions[action] * throttle_step
        dp = self._safe_throttle_move(steer_ + ds, throttle_, dp)
        throttle = max(min(throttle_ + dp, 1.0), 0.0)
        
        dv = dp / dt
        dK = K_max * ds / dt
        v_hat = 0.0 if v == 0 else np.random.normal(v, v * 0.01)

        s_0 = (x, y, theta, x_m_, y_m_, v, K, \
               x_m_hat_, y_m_hat_, theta_hat_, v_hat, \
               dv, dK)

        # integrate to get new state and belief
        I = rk4(self._dsdt, s_0, [0.0, dt])
        x, y, theta, x_m, y_m, v, K, x_m_hat, y_m_hat, theta_hat, v_hat, dv, dK = I[1]
        theta = canonical_angle(theta)
        d += v_hat * dt

        # add noise
        bias = max(-0.01, min(0.01, np.random.normal(bias, pi * 0.0001)))
        theta_hat = canonical_angle(theta + bias)
        v_hat = 0.0 if throttle == 0 else np.random.normal(throttle, throttle * 0.0001) * v_max

        # read sensors
        blueness = self._blueness(x, y)

        # check progress along the track
        x_m, lap, checkpoint = self._median_distance(x, y, x_m_)
        y_m = self._lateral_error(x, y, x_m)
        dx_m = x_m - x_m_
        if lap:
            x_m_hat = 0
            d = 0
        if checkpoint:
            dx_m += lap_median_length
            x_m_hat = -checkpoint_median_length
            d = -checkpoint_median_length

        # lateral position correction from the floor sensor
        if blueness != 0.0:
            y_m_hat = y_m #np.copysign(half_track_width + blue_width * (blueness - 1), y_m_hat_)
    
        # are we done yet?
        out = blueness >= blue_threshold
        wrong_way = self._is_wrong_way(x, y, theta, x_m < 0.0)
        timeout = self.time >= self.time_limit
        done = out or wrong_way or timeout

        # reward progress
        reward = dx_m
        if out or wrong_way:
            reward = self.out_reward

        self.state = (x, y, theta, steer, throttle, x_m, y_m, d)
        self.belief = (x_m_hat, y_m_hat, theta_hat, v_hat, bias)
        if self.show_belief_state:
            observation = np.array((x_m_hat / checkpoint_median_length, y_m_hat / half_track_width, theta_hat / pi, steer, v_hat / v_max))
        else:
            observation = np.array((d / checkpoint_median_length, blueness, theta_hat / pi, steer, v_hat / v_max))

        return observation, reward, done, { \
            'checkpoint': checkpoint,
            'lap': lap,
            'done': 'complete' if timeout else 'out' if out else 'wrong way' if wrong_way else 'unknown'
        }


    def _median_distance(self, x, y, current_mdist):
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
            if current_mdist > - loop_median_length and current_mdist <= line_length:
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
            alpha = np.arctan2(dy, dx) + pi / 2.0
            return line_length + alpha * median_radius, False, False

        # upper-left loop
        else:
            dy = y - median_radius
            dx = -x - median_radius
            alpha = np.arctan2(dx, dy) + pi / 2.0
            return -loop_median_length + alpha * median_radius, False, False


    def _median_to_cartesian(self, x_m, y_m):
        '''
        Calculates the cartesian coordinates of a specific position relative to the track median.
        '''
        # before checkpoint
        if x_m >= 0:
            # lap straight line
            if x_m < line_length:
                return x_m - median_radius, y_m
            # lower-right loop
            else:
                alpha = (x_m - line_length) / median_radius
                x = (median_radius + y_m) * (sin(alpha) + 1)
                y = (median_radius + y_m) * (cos(alpha) - 1)
                return x, y

        # after checkpoint
        else:
            # checkpoint straight line
            if x_m < -loop_median_length:
                return -y_m, x_m + checkpoint_median_length - median_radius
            # upper-left loop
            else:
                alpha = -x_m / median_radius
                x = (y_m - median_radius) * (1 + sin(alpha))
                y = (median_radius - y_m) * (1 - cos(alpha))
                return x, y


    def _median_properties(self, x_m):
        '''
        Calculates the tangent and curvature of a specific positio on the track median.
        '''
        # before checkpoint
        if x_m >= 0:
            # lap straight line
            if x_m < line_length:
                return 0.0, 0.0
            # lower-right loop
            else:
                alpha = (x_m - line_length) / median_radius
                return canonical_angle(-alpha), -1.0 / median_radius / K_max

        # after checkpoint
        else:
            # checkpoint straight line
            if x_m < -loop_median_length:
                return pi / 2.0, 0.0
            # upper-left loop
            else:
                alpha = -x_m / median_radius
                return canonical_angle(-alpha), 1.0 / median_radius / K_max


    def _lateral_error(self, x, y, x_m):
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


    def median_error(self, look_ahead=0.0):
        '''
        Compares the current car position with a trajectory that strictly 
        follows the track median.

        Returns the lateral, heading and steering error.
        '''
        
        x, y, theta, steer, trottle, mdist_ = self.state

        if look_ahead != 0.0:
            mdist = mdist_ + look_ahead
            if mdist > checkpoint_median_length:
                mdist -= lap_median_length
        else:
            mdist = mdist_
        
        theta_, steer_ = self._median_properties(mdist)

        lat_err = self._lateral_error(x, y, mdist_) / half_track_width
        h_err = canonical_angle(theta - theta_) / pi
        st_err = steer - steer_

        return np.array((lat_err, h_err, st_err))


    def _is_wrong_way(self, x, y, theta, checkpoint):
        '''
        Checks if the car is making an illegal turn at the crossing.
        '''
        if abs(x) < half_track_width:
            if y > wrong_way_min and y < wrong_way_max and theta > 0:
                return not checkpoint
            if y < -wrong_way_min and y > -wrong_way_max and theta < 0:
                return True
        elif abs(y) < half_track_width:
            if x > wrong_way_min and x < wrong_way_max and abs(theta) < pi / 2.0:
                return checkpoint
            if x < -wrong_way_min and x > -wrong_way_max and abs(theta) > pi / 2.0:
                return True


    def _safe_throttle_move(self, steer, throttle, desired_change):
        '''
        Moves the throttle by the desired amout or according to the safe speed limit.
        '''
        safe = self._safe_throttle(steer)
        if throttle + desired_change > safe:
            return math.copysign(throttle_step, safe - throttle)
        else:
            return throttle + desired_change

    def _safe_throttle(self, steer):
        '''
        Gets the safe throttle value based on the specified steering.
        '''
        return min(self.throttle_limit, 1.0 - abs(steer) / 2.0)


    def _render(self, mode='human', close=False):
        '''
        Draws the track and the car.
        '''
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            screen_width = 600
            screen_height = 600

            carwidth = 0.12
            carlenght = 0.24

            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(-1.1, 1.1, -1.1, 1.1)

            self.track.set_color(128, 128, 128)
            self.viewer.add_geom(self.track)

            self.breadcrumb = rendering.PolyLine([], close=False)
            self.breadcrumb.set_color(0, 0, 0)
            self.viewer.add_geom(self.breadcrumb)

            l, r, t, b = -carlenght / 2.0, carlenght / 2.0, carwidth / 2.0, -carwidth / 2.0

            self.cartrans = rendering.Transform()

            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.set_color(255, 64, 128)
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)

            d = 0.015
            sensor = rendering.FilledPolygon([(-d, -d), (-d, +d), (+d, +d), (+d, -d)])
            sensor.set_color(255, 0, 0)
            sensor.add_attr(self.cartrans)
            self.viewer.add_geom(sensor)

            steer = rendering.PolyLine([(-0.035, 0.0), (0.035, 0.0)], close=False)
            steer.set_linewidth(3)
            steer.set_color(0, 0, 255)
            self.steertrans = rendering.Transform()
            self.steertrans.set_translation(0.065, 0.0)
            steer.add_attr(self.steertrans)
            steer.add_attr(self.cartrans)
            self.viewer.add_geom(steer)

            carout = rendering.PolyLine([(l, b), (l, t), (r, t), (r, b)], close=True)
            carout.set_linewidth(3)
            carout.set_color(0, 0, 0)
            carout.add_attr(self.cartrans)
            self.viewer.add_geom(carout)
            
        x, y, theta, steer, _, _, _, _ = self.state
        self.cartrans.set_translation(x, y)
        self.cartrans.set_rotation(theta)
        self.steertrans.set_rotation(steer * pi / 2.0)
        self.breadcrumb.v.append((x, y))
        if len(self.breadcrumb.v) > self.trail_length * fps:
            self.breadcrumb.v.pop(0)

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))


    def _track_color(self, x, y):
        '''
        Gets the track color at the specified coordinates.
        '''
        img_x = math.trunc((x + self.track.width / 2) / self.track.width * self.track.img.width)
        img_y = math.trunc((self.track.height - (y + self.track.height / 2)) \
            / self.track.height * self.track.img.height)
        pos = img_x * 4 + img_y * self.track.img.width * 4
        if pos < 0 or pos > len(self.track.img.data):
            return (0, 0, 0, 0)
        else:
            return self.track.img.data[pos:pos + 4]


    def _blueness(self, x, y):
        '''
        Gets the blueness of the track at the specified coordinates.

        The blueness is the normalized difference between the blue and the red 
        channels of the (simulated) RGB color sensor.
        '''
        b, g, r, a = self._track_color(x, y)
        if a == 0:
            return 1.0
        else:
            return (b - r) / 217


def canonical_angle(x):
    '''
    Gets the canonical value of an angle.
    '''
    return wrap(x, -pi, pi)


def wrap(x, m, M):
    """
    :param x: a scalar
    :param m: minimum possible value in range
    :param M: maximum possible value in range
    Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x


def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.
    *y0*
        initial state vector
    *t*
        sample times
    *derivs*
        returns the derivative of the system and has the
        signature ``dy = derivs(yi, ti)``
    *args*
        additional arguments passed to the derivative function
    *kwargs*
        additional keyword arguments passed to the derivative function
    Example 1 ::
        ## 2D system
        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)
    Example 2::
        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)
        y0 = 1
        yout = rk4(derivs, y0, t)
    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),))
    else:
        yout = np.zeros((len(t), Ny))

    yout[0] = y0
    i = 0

    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout
