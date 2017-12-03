import numpy as np

class LowPassFilter(object):
    '''
    First order discrete IIR filter.
    '''
    def __init__(self, feedback_gain, initial_value=0.0):
        self.feedback_gain = np.ones_like(initial_value) * feedback_gain
        self.initial_value = initial_value
        self.output_gain = 1.0 - feedback_gain
        self.input = np.nan
        self.output = initial_value
        self.feedback_value = initial_value / self.output_gain

    def filter(self, value):
        #if not math.isanan(value) and math.isinf(value):
        self.input = value
        self.feedback_value = value + self.feedback_gain * self.feedback_value
        self.output = self.output_gain * self.feedback_value
        
        return self.output

class MovingAverage(object):
    '''
    Moving average filter.
    '''
    def __init__(self, lifetime, sampling_time):
        self.lifetime = lifetime
        self.sampling_time = sampling_time
        self.exp = np.exp(-sampling_time / lifetime)
        self.last_value = None
        self.mean_value = None

    def filter(self, value):
        self.last_value = value
        if self.mean_value is None:
            self.mean_value = value
        else:
            self.mean_value = value + self.exp * (self.mean_value - value)
        
        return self.mean_value
