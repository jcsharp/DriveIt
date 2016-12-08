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

steer_actions = [0.0, 0.1, -0.1]
blue_threshold = 0.75
fps = 60.0
dt = 1.0 / fps
K_max = 4.5
v_max = 2.5

median_radius = 0.375
line_length = 2.0 * median_radius
loop_median_length = 3.0 / 2.0 * pi * median_radius
checkpoint_median_length = line_length + loop_median_length
lap_median_length = 2.0 * checkpoint_median_length

crossing_threshold = 0.225
wrong_way_min = 0.275
wrong_way_max = median_radius

sensor_offeset_x = 0.11
sensor_offeset_y = 0.05

class DriveItEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': fps
    }


    def __init__(self, time_limit=10, throttle_limit=1.0, gamma=0.98, \
                 show_true_mdist=False, trail_length=2.4):
        
        self.time_limit = time_limit
        self.throttle_limit = throttle_limit
        self.show_true_mdist = show_true_mdist
        self.trail_length = trail_length

        # corresponds to the maximum discounted reward over a median lap
        self.max_reward = throttle_limit * v_max * dt / (1 - gamma)

        self.viewer = None

        high = np.array([  1.0,  1.0,  1.0,  1.0 ])
        low  = np.array([ -1.0, -1.0, -1.0, -1.0 ])

        self.action_space = spaces.Discrete(len(steer_actions))
        # for a continuous action space:
        # self.action_space = spaces.Box(self.min_action, self.max_action, shape = (1,))
        self.observation_space = spaces.Box(low, high)

        fname = path.join(path.dirname(__file__), "track.png")
        self.track = rendering.Image(fname, 2., 2.)

        self._seed()
        self.time = -1.0
        self.reset()


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _visibleState(self):
        x, y, theta, steer, throttle, median_distance = self.position
        lap_distance, blue_left, blue_right = self.sensors
        b_rl = blue_right - blue_left
        if self.show_true_mdist:
            return np.array((median_distance / checkpoint_median_length, theta / pi, steer, b_rl))
        else:
            return np.array((lap_distance / checkpoint_median_length, theta / pi, steer, b_rl))


    def _reset(self):
        if self.time == 0.0:
            return self._visibleState()
        
        if self.viewer:
            self.breadcrumb.v.clear()
        
        # random position along the track median
        median_distance = np.random.uniform(-checkpoint_median_length, checkpoint_median_length)
        x, y, theta, steer = self._position_from_median_distance(median_distance)

        # add some noise
        x += np.random.uniform(-0.03, 0.03)
        y += np.random.uniform(-0.03, 0.03)
        theta += np.random.uniform(-pi / 36.0, pi / 36.0)
        steer += steer_actions[np.random.randint(0, 2)]

        # initial sensor values
        blue_left, blue_right, _ = self._sensors_blueness(x, y, cos(theta), sin(theta))
        
        self.time = 0.0
        self.bias = 0.0
        self.position = (x, y, theta, steer, self.safe_throttle(steer), median_distance)
        self.sensors = (median_distance, blue_left, blue_right)
        return self._visibleState()


    def _step(self, action):
        x, y, theta, steer_, throttle_, median_distance_ = self.position
        lap_distance, blue_left_, blue_right_ = self.sensors
        
        self.time += dt

        ds = steer_actions[action]
        steer = max(min(steer_ + ds, 1.0), -1.0)

        dp = self._auto_throttle(steer, throttle_)
        throttle = max(min(throttle_ + dp, 1.0), -1.0)

        # make small random changes to the steering bias
        #self.bias = min(0.1, max(-0.1, np.random.normal(bias, 0.005)))

        # average curvature and speed, with some noise
        K = K_max * ((steer + steer_) / 2.0 + self.bias)
        v = v_max * (throttle + throttle_) / 2.0
        #v = abs(np.random.normal(v, v * 0.01))

        # calculate the new position
        dd = v * dt
        theta = angle_normalize(theta + dd * K)
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        x += dd * cos_theta
        y += dd * sin_theta

        # read sensors
        lap_distance += dd
        blue_left, blue_right, blue_center = self._sensors_blueness(x, y, cos_theta, sin_theta)
        b_rl = blue_right - blue_left
        median_distance, lap, checkpoint = self._median_distance(x, y, median_distance_)
        ddist = median_distance - median_distance_

        if lap:
            lap_distance = 0

        if checkpoint:
            ddist += lap_median_length
            lap_distance = -checkpoint_median_length

        # are we done yet?
        out = blue_center >= blue_threshold
        wrong_way = self._is_wrong_way(x, y, theta, median_distance < 0.0)
        timeout = self.time >= self.time_limit
        done = out or wrong_way or timeout

        # reward progress
        reward = ddist
        # discourage over-steering
        #reward -= abs(ds) * dt if ds * steer >= 0.0 else 0.0
        # penalty for getting closer to the track boundary
        #Sreward -= (blue_left - blue_left_ + blue_right - blue_right_) / 2.0 * dt
        #reward -= (blue_left + blue_right) / 2.0 * dt
        # do we need further punishment when we exit the tracks?
        if out or wrong_way:
            reward = -self.max_reward

        self.position = (x, y, theta, steer, throttle, median_distance)
        self.sensors = (lap_distance, blue_left, blue_right)
        state = self._visibleState()
        return state, reward, done, { \
            'checkpoint': checkpoint,
            'lap': lap,
            'done': 'complete' if timeout else 'out' if out else 'wrong way' if wrong_way else 'unknown'
        }


    def _median_distance(self, x, y, current_mdist):
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


    def _position_from_median_distance(self, mdist):
        # before checkpoint
        if mdist >= 0:
            # lap straight line
            if mdist < line_length:
                return mdist - median_radius, 0.0, 0.0, 0.0
            # lower-right loop
            else:
                alpha = (mdist - line_length) / median_radius
                x = median_radius * (sin(alpha) + 1)
                y = median_radius * (cos(alpha) - 1)
                return x, y, angle_normalize(-alpha), -median_radius / K_max

        # after checkpoint
        else:
            # checkpoint straight line
            if mdist < -loop_median_length:
                return 0.0, mdist + checkpoint_median_length - median_radius, pi / 2.0, 0.0
            # upper-left loop
            else:
                alpha = -mdist / median_radius
                x = -median_radius * (1 + sin(alpha))
                y = median_radius * (1 - cos(alpha))
                return x, y, angle_normalize(-alpha), median_radius / K_max


    def median_error(self, look_ahead=0.05):
        
        x, y, theta, steer, trottle, mdist_ = self.position
        mdist = mdist_ + look_ahead
        if mdist > checkpoint_median_length:
            mdist -= lap_median_length

        _, _, theta_, steer_ = self._position_from_median_distance(mdist)

        # before checkpoint
        if mdist >= 0:
            # lap straight line
            if mdist < line_length:
                lat_err = y

            # lower-right loop
            else:
                dx = x - median_radius
                dy = y + median_radius
                lat_err = math.sqrt(dx ** 2 + dy ** 2) - median_radius

        # after checkpoint
        else:
            # checkpoint straight line
            if mdist < -loop_median_length:
                lat_err = -x

            # upper-left loop
            else:
                dx = x + median_radius
                dy = y - median_radius
                lat_err = median_radius - math.sqrt(dx ** 2 + dy ** 2) 

        lat_err /= 0.225
        h_err = angle_normalize(theta - theta_) / pi
        st_err = steer - steer_

        return np.array((lat_err, h_err, st_err))


    def _is_wrong_way(self, x, y, theta, checkpoint):
        if abs(x) < crossing_threshold:
            if y > wrong_way_min and y < wrong_way_max and theta > 0:
                return not checkpoint
            if y < -wrong_way_min and y > -wrong_way_max and theta < 0:
                return True
        elif abs(y) < crossing_threshold:
            if x > wrong_way_min and x < wrong_way_max and abs(theta) < pi / 2.0:
                return checkpoint
            if x < -wrong_way_min and x > -wrong_way_max and abs(theta) > pi / 2.0:
                return True


    def _auto_throttle(self, steer, throttle):
        safe = self.safe_throttle(steer)
        return math.copysign(dt * 2.5, safe - throttle)


    def safe_throttle(self, steer):
        return min(self.throttle_limit, 1.0 - abs(steer) / 2.0)


    def _render(self, mode='human', close=False):
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

            x = sensor_offeset_x
            y = sensor_offeset_y
            d = 0.01
            sensor = rendering.FilledPolygon([(x-d, y-d), (x-d, y+d), (x+d, y+d), (x+d, y-d)])
            sensor.set_color(255, 0, 0)
            sensor.add_attr(self.cartrans)
            self.viewer.add_geom(sensor)
            sensor = rendering.FilledPolygon([(x-d, -y-d), (x-d, -y+d), (x+d, -y+d), (x+d, -y-d)])
            sensor.set_color(255, 0, 0)
            sensor.add_attr(self.cartrans)
            self.viewer.add_geom(sensor)

            carout = rendering.PolyLine([(l, b), (l, t), (r, t), (r, b)], close=True)
            carout.set_linewidth(3)
            carout.set_color(0, 0, 0)
            carout.add_attr(self.cartrans)
            self.viewer.add_geom(carout)
            
            #self.floorWin = rendering.PolyLine(\
            #    [(-flcw,-flcw), (-flcw,flcw), (flcw,flcw), (flcw,-flcw)], close=True)
            #self.floorWin.set_linewidth(3)
            #self.floorWin.set_color(255,0,0)
            #self.floorWin.add_attr(self.cartrans)
            #self.viewer.add_geom(self.floorWin)

        x, y, theta, _, _, _ = self.position
        self.cartrans.set_translation(x, y)
        self.cartrans.set_rotation(theta)
        #b,g,r,a = self._track_color(x,y)
        #self.floorWin.set_color(r,g,b)
        self.breadcrumb.v.append((x, y))
        if len(self.breadcrumb.v) > self.trail_length * fps:
            self.breadcrumb.v.pop(0)

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))


    def _track_color(self, x, y):
        img_x = math.trunc((x + self.track.width / 2) / self.track.width * self.track.img.width)
        img_y = math.trunc((self.track.height - (y + self.track.height / 2)) \
            / self.track.height * self.track.img.height)
        pos = img_x * 4 + img_y * self.track.img.width * 4
        if pos < 0 or pos > len(self.track.img.data):
            return (0, 0, 0, 0)
        else:
            return self.track.img.data[pos:pos + 4]


    def _blueness(self, x, y):
        b, g, r, a = self._track_color(x, y)
        if a == 0:
            return 1.0
        else:
            return (b - r) / 255


    def _sensors_blueness(self, x, y, cos_theta, sin_theta):
        offset_x_l = sensor_offeset_x * cos_theta - sensor_offeset_y * sin_theta
        offset_x_r = sensor_offeset_x * cos_theta + sensor_offeset_y * sin_theta
        offset_y_l = sensor_offeset_x * sin_theta + sensor_offeset_y * cos_theta
        offset_y_r = sensor_offeset_x * sin_theta - sensor_offeset_y * cos_theta
        blue_left = self._blueness(x + offset_x_l, y + offset_y_l)
        blue_right = self._blueness(x + offset_x_r, y + offset_y_r)
        return (blue_left, blue_right, self._blueness(x, y)) #(blue_left + blue_right) / 2.0)


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
