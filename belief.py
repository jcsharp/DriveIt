from DriveItMultiGym import *
from car import *
from gym import spaces
from DriveItCircuit import *



class BeliefDriveItEnv(DriveItEnv):

    def __init__(self, car=Car(), agents=list(), time_limit=10, noisy=True):
        self.tracker = PositionTracking(car)
        super().__init__(car, agents, time_limit, noisy)
        self.observation_space = self.tracker.observation_space

    def _reset(self, random_position=True):
        obs = super()._reset(random_position)
        return self.tracker.normalize(self.tracker.reset(obs))

    def _step(self, action):
        obs, reward, done, info = super()._step(action)
        b = self.tracker.update(obs, self.dt)
        return self.tracker.normalize(b), reward, done, info


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

        x_m, y_m, theta_m = cartesian_to_median(x, y, theta)

        def print_adjustment(name, value, new_value, real_value, previous):
            change = new_value - value
            desired = real_value - value
            error = desired - change
            print('%s adjusted by %f (real %f, err %f)' % (name, change, desired, error))

        pos_adjusted = False

        # checkpoint threshold
        if checkpoint and not checkpoint_:
            if x_m > 0.0:
                #print_adjustment('y>', y, -half_track_width, self.car.get_position()[1], y_)
                x_m = -checkpoint_median_length
                y = -half_track_width
                pos_adjusted = True

        # lap threshold
        elif checkpoint_ and not checkpoint:
            if x_m < 0.0:
                #print_adjustment('x>', x, -half_track_width, self.car.get_position()[0], x_)
                x_m = 0
                x = -half_track_width
                pos_adjusted = True
        
        elif checkpoint and x_m > 0.0:
            #print_adjustment('x<', x, -half_track_width, self.car.get_position()[0], x_)
            x_m = 0.0
            x = -half_track_width
            pos_adjusted = True
        
        elif x_m > checkpoint_median_length:
            #print_adjustment('y<', y, -half_track_width, self.car.get_position()[1], y_)
            x_m = checkpoint_median_length
            y = -half_track_width
            pos_adjusted = True

        if pos_adjusted:
            x_m, y_m, theta_m = cartesian_to_median(x, y, theta)

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
