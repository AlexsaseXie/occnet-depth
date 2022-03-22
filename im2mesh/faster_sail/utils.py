import numpy as np
import torch

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

def find_r_t(points):
    '''
        SAIL-S3: find r, t
        Input: points (n * 3) np array
    '''
    n = points.shape[0]
    A = np.concatenate([points * 2, np.ones((n,1))], axis=1) # n * 4
    y = (points * points).sum(axis=1)
    b = np.linalg.inv(A.T @ A) @ A.T @ y
    t = b[0:3]
    r = np.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2] + b[3])

    return r, t
