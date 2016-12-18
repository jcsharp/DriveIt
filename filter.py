import numpy as np

class LowPassFilter():
    '''
    First order discrete IIR filter.
    '''
    def __init__(self, feedback_gain, initial_value=0.0):
        self.feedback_gain = np.ones_like(initial_value) * feedback_gain
        self.initial_value = initial_value
        self.output_gain = 1.0 - feedback_gain
        self.output = initial_value
        self.feedback_value = initial_value / self.output_gain

    def filter(self, value):
        #if not math.isanan(value) and math.isinf(value):
        self.input = value
        self.feedback_value = value + self.feedback_gain * self.feedback_value
        self.output = self.output_gain * self.feedback_value
        
        return self.output
