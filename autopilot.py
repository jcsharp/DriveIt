# -*- coding: utf-8 -*-
"""
Autopilots for the DriveIt Gym environment.
@author: Jean-Claude Manoli
"""
import numpy as np
from belief import BeliefTracking
from PositionTracking import PositionTracking

epsilon = 0.05


class Autopilot(object):
    def __init__(self, car, other_cars=None, tracker_type=PositionTracking):
        self.car = car
        self.tracker = BeliefTracking(car, other_cars, tracker_type, normalize=True)
        self.belief, self.deltas = [], []
        self.action = 0

    def reset(self, x_m, observation):
        belief = self.tracker.reset(x_m, observation)
        self.deltas = np.zeros(np.shape(belief))
        self.belief = belief
        return belief

    def observe(self, observation, dt):
        belief = self.tracker.update(self.action, observation, dt)
        self.deltas = belief - self.belief
        self.belief = belief
        return belief

    def act(self):
        self.action = self._act()
        return self.action

    def _act(self): raise NotImplementedError


class LookAheadPilot(Autopilot):
    def __init__(self, car, other_cars=None, tracker_type=PositionTracking,
                 ky=3.0, kdy=10.0, 
                 kth=3.0, kdth=10.0, 
                 kka=3.0, kdka=-3.0):
        super().__init__(car, other_cars, tracker_type)
        self.params = ky, kdy, kth, kdth, kka, kdka

    def _danger(self, dist, ddist, x):
        d, dd, ddd, yi = False, False, 1.0, False
        if x < 0.0 and x > -1.0:
            yi = True
            if dist[0] < 0.6:
                d, dd = True, True
        for i in range(0, min(3, len(dist))):
            ddd = min(ddd, ddist[i])
            if dist[i] < (0.4 if i == 0 else 0.95):
                d = True
                if yi or dist[i] < (0.25 if i == 0 else 0.6):
                    dd = True
                
        return d, dd, ddd, yi

    def _act(self):
        x, y, th, v, k, kt, ka, *dist = self.belief #pylint: disable=W0612
        dx, dy, dth, dv, dk, dkt, dka, *ddist = self.deltas #pylint: disable=W0612
        ky, kdy, kth, kdth, kka, kdka = self.params

        fy = ky * y + kdy * dy
        fth = kth * th + kdth * dth
        fk = kka * (ka - k) + kdka * (dka - k)
        f = -fy + fth + fk - k
        if f > epsilon: action = 1
        elif f < -epsilon: action = 2
        else: action = 0
        
        d, dd, ddd, yi = self._danger(dist, ddist, x)
        safe_throttle = self.car.specs.safe_turn_speed( \
            max(abs(k), abs(ka)), 0.9) / self.car.specs.v_max
        if (not d or d and not yi and ddd >= 0.0) and v < safe_throttle - epsilon:
            action += 3
        elif dd or (d and ddd < 0.0) or v > safe_throttle + epsilon:
            action += 6

        return action
