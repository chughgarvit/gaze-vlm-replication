"""Bounded unsupervised gaze calibration; wearer identity is required for claims."""
import numpy as np


class GazeCalibration:
    def __init__(self, population_mean, population_std, prior_samples=256, max_samples=1024):
        self.population_mean = np.asarray(population_mean, dtype=np.float64)
        self.population_std = np.maximum(np.asarray(population_std, dtype=np.float64), .05)
        self.prior_samples, self.max_samples = prior_samples, max_samples
        self.samples = []

    def transform(self, gaze):
        result = gaze.copy()
        valid = result[:, 3] > 0
        if self.samples:
            observed = np.asarray(self.samples)
            weight = len(observed)/(len(observed)+self.prior_samples)
            mean = (1-weight)*self.population_mean + weight*observed.mean(0)
            std = (1-weight)*self.population_std + weight*np.maximum(observed.std(0), .05)
        else:
            mean, std = self.population_mean, self.population_std
        result[valid, 1:3] = (result[valid, 1:3]-mean)/std
        return result

    def observe(self, gaze):
        self.samples.extend(gaze[gaze[:, 3] > 0, 1:3].tolist())
        self.samples = self.samples[-self.max_samples:]
