# -*- coding: utf-8 -*-
"""
Autopilots for the DriveIt Gym environment.
@author: Jean-Claude Manoli
"""
import numpy as np
from belief import BeliefTracking
from PositionTracking import PositionTracking, TruePosition

epsilon = 0.05


class Autopilot(object):
    def __init__(self, car, other_cars=None, tracker_type=PositionTracking):
        self.car = car
        self.tracker = BeliefTracking(car, other_cars, tracker_type, normalize=True)
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


def _danger(dist, ddist, x):
    d, dd, ddd, yi = False, False, 1.0, False
    if x < 0.0 and x > -0.2:
        yi = True
        if dist[0] < 0.6:
            d, dd = True, True
    for i in range(0, 3 if len(dist) > 2 else 1):
        ddd = min(ddd, ddist[i])
        if dist[i] < (0.5 if i == 0 else 0.95):
            d = True
            if yi or dist[i] < (0.35 if i == 0 else 0.7):
                dd = True
            
    return d, dd, ddd, yi


def LeftLaneFollowingPilot(car, other_cars):
    return LaneFollowingPilot(car, other_cars, TruePosition, 0.5)

def RightLaneFollowingPilot(car, other_cars):
    return LaneFollowingPilot(car, other_cars, TruePosition, -0.5)

class LaneFollowingPilot(Autopilot):
    def __init__(self, car, other_cars=None, tracker_type=TruePosition,
                 offset=0.0, ky=10, kdy=100, kka=6):
        super().__init__(car, other_cars, tracker_type)
        self.car.specs.set_lateral_offset(offset)
        self.params = ky, kdy, kka

    def _act(self):
        x, y, th, v, k, kt, ka, *dist = self.belief #pylint: disable=W0612
        dx, dy, dth, dv, dk, dkt, dka, *ddist = self.deltas #pylint: disable=W0612
        ky, kdy, kka = self.params
        offset = self.car.specs.lateral_offset

        f = ky * (offset - y) - kdy * dy + kka * (ka - k)
        steer = np.clip(f, -1.0, 1.0)

        d, dd, ddd, yi = _danger(dist, ddist, x)
        throttle = self.car.specs.safe_turn_speed( \
            max(abs(k), abs(ka)), 0.9) / self.car.specs.v_max

        if (not d or d and not yi and (ddd >= 0.0 or v < 0.05)) and v < throttle - epsilon:
            throttle = min(v + 0.1, 1.0)
        elif dd or (d and ddd < 0.0) or v > throttle + epsilon:
            throttle = max(v - 0.1, 0.0)

        return steer, throttle


def ReflexPilot(car, other_cars):
    return LookAheadPilot(car, other_cars, TruePosition, kka=1.0, kdka=1.0)

def SharpPilot(car, other_cars):
    return LookAheadPilot(car, other_cars, TruePosition)

class LookAheadPilot(Autopilot):
    def __init__(self, car, other_cars=None, tracker_type=PositionTracking,
                 ky=3.0, kdy=10.0, 
                 kth=3.0, kdth=10.0, 
                 kka=3.0, kdka=-3.0):
        super().__init__(car, other_cars, tracker_type)
        self.params = ky, kdy, kth, kdth, kka, kdka

    def _act(self):
        x, y, th, v, k, kt, ka, *dist = self.belief #pylint: disable=W0612
        dx, dy, dth, dv, dk, dkt, dka, *ddist = self.deltas #pylint: disable=W0612
        ky, kdy, kth, kdth, kka, kdka = self.params

        fy = ky * y + kdy * dy
        fth = kth * th + kdth * dth
        fk = kka * (ka - k) + kdka * (dka - k)
        f = -fy + fth + fk - k
        if f > epsilon: steer = min(k + 0.1, 1.0)
        elif f < -epsilon: steer = max(k - 0.1, -1.0)
        else: steer = k

        d, dd, ddd, yi = _danger(dist, ddist, x)
        throttle = self.car.specs.safe_turn_speed( \
            max(abs(k), abs(ka)), 0.9) / self.car.specs.v_max

        if (not d or d and not yi and (ddd >= 0.0 or v < 0.05)) and v < throttle - epsilon:
            throttle = min(v + 0.1, 1.0)
        elif dd or (d and ddd < 0.0) or v > throttle + epsilon:
            throttle = max(v - 0.1, 0.0)

        return steer, throttle
