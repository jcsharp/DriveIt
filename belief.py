from DriveItMultiGym import *
from car import *
from gym import spaces
from DriveItCircuit import *

class PositionTracking():

    def __init__(self, car=Car()):
        self.car = car
        high = np.array([  checkpoint_median_length,  half_track_width,  pi, car.specs.v_max,  car.specs.K_max ])
        low  = np.array([ -checkpoint_median_length, -half_track_width, -pi,             0.0, -car.specs.K_max ])
        self.action_space = spaces.Discrete(len(steer_actions))
        self.observation_space = spaces.Box(low, high)


    def reset(self, observation):
        d, blue, theta, v, K, thres = observation
        x, y, _ = median_to_cartesian(d, 0.0, 0.0)
        self.observation = observation
        self.position = x, y, d < 0.0
        return d, 0., theta, v, K


    def update(self, observation, dt):
        x_, y_, checkpoint_ = self.position
        d, blue, theta, v, K, checkpoint = observation
        d_, blue_, theta_, v_, K_, _ = self.observation
        self.observation = observation

        a = (v - v_) / dt
        K_dot = (K - K_) / dt
        x, y, _, _, _, _ = Car._move(x_, y_, theta_, v_, K_, d_, a, K_dot, dt)

        x_m, y_m, theta_m = cartesian_to_median(x, y, theta, checkpoint)

        pos_adjusted = False

        #if d == -checkpoint_median_length: # checkpoint
        if checkpoint and not checkpoint_: # checkpoint threshold
            #checkpoint = True
            if x_m > 0.0:
                #print('y adjusted chkp %f', (-median_radius - y))
                x_m = -checkpoint_median_length
                y = -median_radius
                pos_adjusted = True
        #elif d == 0.0: # lap
        elif checkpoint_ and not checkpoint: # lap threshold
            #checkpoint = False
            if x_m < 0.0:
                #print('x adjusted lap %f', (-median_radius - x))
                x_m = 0
                x = -median_radius
                pos_adjusted = True
        
        if checkpoint and x_m > 0.0:
            x_m = 0.0
            x = -median_radius
            pos_adjusted = True
            #print('x adjusted')
        
        if x_m > checkpoint_median_length:
            x_m = checkpoint_median_length
            y = -median_radius
            pos_adjusted = True
            #print('y adjusted')

        if pos_adjusted:
            x_m, y_m, theta_m = cartesian_to_median(x, y, theta, checkpoint)

        self.position = (x, y, checkpoint)

        return x_m, y_m, theta_m, v, K


    def normalize(self, belief):
        return belief / self.observation_space.high


    def reset_all(trackers, observations):
        b = {}
        for tracker in trackers:
            b[tracker.car] = tracker.reset(observations[tracker.car])
        return b


    def update_all(trackers, observations, dt):
        b = {}
        for tracker in trackers:
            b[tracker.car] = tracker.update(observations[tracker.car], dt)
        return b
