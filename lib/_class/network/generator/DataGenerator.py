import numpy as np

from tensorflow.keras.utils import Sequence

# Reference: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(Sequence):
    def __init__(self, inputs, targets, batch_size):
        self.inputs     = inputs
        self.targets    = targets
        self.batch_size = batch_size
        
    def __len__(self):
        return int(np.ceil(len(self.inputs) / self.batch_size))
    
    def __getitem__(self, index):
        batch_inputs  = self.inputs [index * self.batch_size : (index + 1) * self.batch_size]
        batch_targets = self.targets[index * self.batch_size : (index + 1) * self.batch_size]
        
        return batch_inputs, batch_targets