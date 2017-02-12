import numpy as np
import math, random
from brain import Brain
from Memory import SumTreeMemory

MAX_EPSILON = 1.0
MIN_EPSILON = 0.1 # stay a bit curious even when getting old
EXPLORATION_STOP = 3000000   # at this step epsilon will be MIN_EPSILON
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay

MEMORY_CAPACITY = 5000000
BATCH_SIZE_PER_SAMPLE = 8
UPDATE_TARGET_FREQUENCY = 10000
GAMMA = 0.995 # discount factor

class Agent:
    steps = 0
    episodes = 0
    epsilon = MAX_EPSILON
    lastTargetUpdateStep = 0
    episode = []

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        
        self.brain = Brain(stateCnt, actionCnt)
        self.memory = SumTreeMemory(MEMORY_CAPACITY)
        
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt - 1)
        else:
            return np.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.episode.insert(0, sample)
        self.steps += 1

    def endEpisode(self):
        n = len(self.episode)
        if n == 0: return

        # calculate Q_min values for the episode steps
        q = 0
        batch = []
        for i in range(n):
            s, a, r, s_ = self.episode[i]
            q = r + GAMMA * q
            batch.append((s, a, r, s_, q)) 
        
        # get the target model error on the steps for prioritized replay 
        x, y, err = self._getTargets(batch)
        
        # add steps to memory, prioritized by error
        for i in range(len(batch)):
            self.memory.add(err[i], batch[i])
        
        self.episode = []
        self.episodes += 1

        # slowly decrease epsilon based on our eperience
        self.epsilon = MIN_EPSILON \
        + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)
        
        # memory replay based on the number of steps in the episode
        self.replay(int(n * BATCH_SIZE_PER_SAMPLE))
        
    def _getTargets(self, batch):
        no_state = np.zeros(self.stateCnt)
        states = np.array([ o[0] for o in batch ], dtype=np.float32)
        states_ = np.array([ (no_state if o[3] is None else o[3]) \
                            for o in batch ], dtype=np.float32)

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_, target = False)
        pTarget_ = self.brain.predict(states_, target = True)
        
        x = np.zeros((len(batch), self.stateCnt)).astype(np.float32)
        y = np.zeros((len(batch), self.actionCnt)).astype(np.float32)
        err = np.zeros(len(batch))

        for i in range(len(batch)):
            s, a, r, s_, q = batch[i]
           
            t = p[i,0]
            if s_ is None:
                ta = r
            else:
                #ta = r + GAMMA * pTarget_[i,0][ np.argmax(p_[i,0]) ]  # double DQN
                ta = max(q, r + GAMMA * pTarget_[i,0][ np.argmax(p_[i,0]) ]) # Q-min DDQN

            err[i] = np.abs(t[a] - ta)
            t[a] = ta
            x[i] = s
            y[i] = t
            
        return (x, y, err)
        
    def replay(self, size):
        # eriodically update the target network
        if self.steps - self.lastTargetUpdateStep >= UPDATE_TARGET_FREQUENCY:
            self.brain.updateTargetModel()
            self.lastTargetUpdateStep = self.steps
        
        # get samples batch and train over the targets
        indexes, batch = self.memory.sample(size)
        x, y, err = self._getTargets(batch)
        self.brain.train(y, x)
        
        # update the batch steps errors in memory
        for i in range(len(batch)):
            self.memory.update(indexes[i], err[i])
