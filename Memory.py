class Memory:
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, observation):
        self.samples.append(observation)
        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)
    
    def isFull(self):
        return len(self.samples) >= self.capacity
    
    def length(self):
        return len(self.samples)    
    
from SumTree import SumTree
from numpy import random

class SumTreeMemory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    count = 0

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, err):
        return (err + self.e) ** self.a

    def add(self, err, sample):
        p = self._getPriority(err)
        self.tree.add(p, sample) 
        self.count += 1

    def sample(self, n):
        batch = []
        indexes = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append(data)
            indexes.append(idx)

        return indexes, batch

    def update(self, idx, err):
        p = self._getPriority(err)
        self.tree.update(idx, p)
        
    def isFull(self):
        return self.count >= self.tree.capacity
        
    def length(self):
        return self.count