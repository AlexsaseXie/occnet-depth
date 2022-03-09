import numpy as np

def adjust_learning_rate(optimizer, new_lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = new_lr

class StepLearningRateSchedule:
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return np.maximum(self.initial * (self.factor ** (epoch // self.interval)),1.0e-5)
