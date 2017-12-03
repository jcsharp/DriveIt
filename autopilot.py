# -*- coding: utf-8 -*-
"""
Autopilots for the DriveIt Gym environment.
@author: Jean-Claude Manoli
"""
import numpy as np
from belief import BeliefTracking

epsilon = 0.05

class Autopilot(object):
    def __init__(self, car, other_cars=None):
        self.car = car
        self.tracker = BeliefTracking(car, other_cars, normalize=True)
        self.belief, self.deltas = [], []
        self.action = 0

    def reset(self, observation):
        belief = self.tracker.reset(observation)
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
    def __init__(self, car, other_cars=None, 
                 ky=3.0, kdy=10.0, 
                 kth=3.0, kdth=10.0, 
                 kka=3.0, kdka=-3.0):
        super().__init__(car, other_cars)
        self.params = ky, kdy, kth, kdth, kka, kdka

    def _danger(self, dist):
        for d in dist[:3]:
            if d < 0.18 \
            or d < 0.5 and self.tracker.observation[-1]:
                return True
        return False

    def _act(self):
        y, th, v, k, kt, ka, *dist = self.belief #pylint: disable=W0612
        dy, dth, dv, dk, dkt, dka, *ddist = self.deltas #pylint: disable=W0612
        ky, kdy, kth, kdth, kka, kdka = self.params

        fy = ky * y + kdy * dy
        fth = kth * dth + kdth * dth
        fk = kka * (ka - k) + kdka * (dka - k)
        f = -fy + fth + fk - k
        if f > epsilon: action = 1
        elif f < -epsilon: action = 2
        else: action = 0
        
        if self._danger(dist):
            action += 6
        else:
            safe_throttle = self.car.specs.safe_turn_speed( \
                max(abs(k), abs(ka)), 0.9) / self.car.specs.v_max
            if v < safe_throttle - epsilon:
                action += 3
            elif v > safe_throttle + epsilon:
                action += 6

        return action
