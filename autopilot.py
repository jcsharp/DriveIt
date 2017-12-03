import numpy as np

epsilon = 0.05

class Autopilot(object):
    def __init__(self, car,
                 ky=3.0, kdy=10.0, 
                 kth=3.0, kdth=10.0, 
                 kka=3.0, kdka=-3.0):
        self.car = car
        self.observation, self.deltas = [], []
        self.params = ky, kdy, kth, kdth, kka, kdka

    def reset(self, observation):
        self.observation = observation
        self.deltas = np.zeros(np.shape(observation))

    def observe(self, observation):
        self.deltas = observation - self.observation
        self.observation = observation

    def act(self):
        y, th, v, k, kt, ka, d1, d2, d3, d4, d5 = self.observation
        dy, dth, dv, dk, dkt, dka, dd1, dd2, dd3, dd4, dd5 = self.deltas
        ky, kdy, kth, kdth, kka, kdka = self.params

        fy = ky * y + kdy * dy
        fth = kth * dth + kdth * dth
        fk = kka * (ka - k) + kdka * (dka - k)
        f = -fy + fth + fk - k
        if f > epsilon: action = 1
        elif f < -epsilon: action = 2
        else: action = 0
        
        safe_throttle = self.car.specs.safe_turn_speed( \
            max(abs(k), abs(ka)), 0.9) / self.car.specs.v_max
        if v < safe_throttle - epsilon:
            action += 3
        elif v > safe_throttle + epsilon:
            action += 6

        return action
