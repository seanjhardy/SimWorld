# SOURCE: adapted from https://github.com/cokwa/bitHTM

import numpy as np


class NoBoosting:
    def process(self, input):
        return input
    def update(self, columns):
        return None

class ExponentialBoosting:
    def __init__(
        self, output_dim, sparsity,
        intensity=0.2, momentum=0.99
    ):
        self.sparsity = sparsity
        self.intensity = intensity
        self.momentum = momentum

        self.duty_cycle = np.zeros(output_dim, dtype=np.float32)

    def process(self, input_activation):
        factor = np.exp(-(self.intensity / self.sparsity) * self.duty_cycle)
        return factor * input_activation

    def update(self, active_input):
        self.duty_cycle *= self.momentum
        self.duty_cycle[active_input] += 1.0 - self.momentum


class UpperLowerThresholdBoosting:
    def __init__(
        self, output_dim, sparsity,
        intensity=0.2, momentum=0.99,
        lower_threshold=0.001, upper_threshold=0.3
    ):
        self.sparsity = sparsity
        self.intensity = intensity
        self.momentum = momentum
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

        self.duty_cycle = np.zeros(output_dim, dtype=np.float32)
        self.active = np.ones(output_dim, dtype=np.bool_)

    def process(self, input_activation):
        factor = np.exp(-(self.intensity / self.sparsity) * self.duty_cycle)
        #Activate neurons above threshold and deactivate neurons below the thresold
        active = np.where((self.active == 1) & (factor < self.lower_threshold), 0, self.active)
        self.active = np.where((self.active == 0) & (factor > self.upper_threshold), 1, active)
        return self.active

    def update(self, active_input):
        self.duty_cycle *= self.momentum
        self.duty_cycle[active_input] += 1.0 - self.momentum
        
class SoftmaxInhibition:
    def __init__(self, threshold=0.05):
        self.threshold = threshold
    
    def process(self, input):
        nonzero = np.nonzero(input)[0]

        softmax = np.zeros((len(input)))
        softmax[nonzero] = np.exp(input[nonzero]) / np.sum(np.exp(input[nonzero]))
        sorted = softmax.argsort()[::-1]
        total = 0
        active_columns = []
        while total < self.threshold and len(sorted) > 0:
            active_columns.append(sorted[0])
            total += sorted[0]
            sorted = sorted[1:]

        return active_columns
        
class GlobalInhibition:
    def __init__(self, sparsity):
        self.sparsity = sparsity

    def process(self, input_activation, k_winners=None):
        k = k_winners or int(len(input_activation) * self.sparsity)
        return np.argpartition(input_activation, -k)[-k:]

class LocalInhibition:
    def __init__(self, sparsity):
        self.sparsity = sparsity

    def process(self, input):
        k_winners = int(len(input) * self.sparsity)
        # Add padding to be buckets of size k_winners
        padding = int(np.ceil(len(input) / k_winners) * k_winners) - len(input)
        input = np.pad(input, int(padding/2), mode="constant")
        # Split data into buckets
        window = np.array_split(input, k_winners)
        # Compute max indices in each bucket
        active = np.argmax(window, axis=1)
        # Add back correct starting indices
        active += (np.arange(0, k_winners, 1) * 1.0/self.sparsity).astype(np.int32)
        return active