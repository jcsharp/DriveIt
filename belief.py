from DriveItMultiGym import *
from car import *

class PositionTracking():

    def __init__(self, specs=CarSpecifications()):
        self.specs = specs


    def reset(self, observation):
        d, blue, theta, v, K = observation
        x, y = DriveItEnv.median_to_cartesian(d, 0.0)
        self.observation = observation
        self.position = x, y, d < 0.0
        return d, 0.0, theta, v, K, blue


    def update(self, observation, dt):
        x_, y_, checkpoint = self.position
        d, blue, theta, v, K = observation
        d_, blue_, theta_, v_, K_ = self.observation
        self.observation = observation

        a = (v - v_) / dt
        K_dot = (K - K_) / dt
        x, y, _, _, _, _ = Car._move(x_, y_, theta_, v_, K_, d_, a, K_dot, dt)

        x_m, _, _ = DriveItEnv.median_distance(x, y, d_)

        if d == -checkpoint_median_length: # checkpoint
            checkpoint = True
            if x_m > 0.0:
                x_m = -checkpoint_median_length
                y = -median_radius
        elif d == 0.0: # lap
            checkpoint = False
            if x_m < 0.0:
                x_m = 0
                x = -median_radius
        
        if checkpoint and x_m > 0.0:
            x_m = 0.0
            x = -median_radius
        
        if x_m > checkpoint_median_length:
            x_m = checkpoint_median_length
            y = -median_radius

        y_m = DriveItEnv.lateral_error(x, y, x_m)

        self.position = (x, y, checkpoint)

        return x_m, y_m, theta, v, K, blue


    def reset_all(trackers, observations):
        b = []
        for i in range(len(trackers)):
            b.append(trackers[i].reset(observations[i]))
        return b


    def update_all(trackers, observations, dt):
        b = []
        for i in range(len(trackers)):
            b.append(trackers[i].update(observations[i], dt))
        return b
