import numpy as np
from cntk import *
from cntk.models import Sequential
from cntk.layers import *

HL_SIZE = 128

class Brain:
    def __init__(self, state_shape, action_shape):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self._createModel()
        self.updateTargetModel()
        
    def _createModel(self):
        observation = input_variable(self.state_shape, np.float32, name="s")
        q_target = input_variable(self.action_shape, np.float32, name="q")

        l1 = Dense(HL_SIZE)
        l2 = Dense(HL_SIZE, activation=relu)
        l3 = Dense(HL_SIZE, activation=relu)
        #lo = Dense(self.actionCnt)
        #q = Sequential([l1, l2, l3, lo])
        
        v1 = Dense(HL_SIZE, activation=relu)
        vo = Dense(1)
        value = Sequential([l1, l2, l3, v1, vo])
        
        a1 = Dense(HL_SIZE, activation=relu)
        ao = Dense(self.action_shape)
        advantage = Sequential([l1, l2, l3, a1, ao])

        q = plus(value, minus(advantage, reduce_mean(advantage, axis=0)))
        
        self.value = value(observation)
        self.advantage = advantage(observation)
        self.model = q(observation)

        self.loss = reduce_mean(square(self.model - q_target), axis=0)
        eval_function = reduce_mean(square(self.model - q_target), axis=0)

        #     0.0  0.3  0.6   0.9   1.2   1.5
        #lr = [0.5, 0.2, 0.1, 0.05, 0.02, 0.02]
        #epoch_size = 300000 * 8
        #lr_schedule = learning_rate_schedule(lr, UnitType.sample, epoch_size=epoch_size)
        lr_schedule = learning_rate_schedule(0.1, UnitType.minibatch)
        momentum = momentum_schedule(0.9)

        learner = momentum_sgd(self.model.parameters, lr_schedule, momentum, \
                               gradient_clipping_threshold_per_sample=10)
        self.trainer = Trainer(self.model, (self.loss, eval_function), learner)
    
    def train(self, x, y):
        arguments = dict(zip(self.loss.arguments, [y,x]))
        return self.trainer.train_minibatch(arguments, outputs=[self.loss.output])

    def predict(self, s, target=False):
        if target:
            return self.model_.eval(s)
        else:
            return self.model.eval(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.state_shape), target=target).flatten()

    def updateTargetModel(self):
        self.model_ = self.model.clone(method='clone')
        
    def saveModel(self, path):
        self.model.save(path, False)
        
    def loadModel(self, path):
        self.model = load_model(path)
        self._createTrainer()
