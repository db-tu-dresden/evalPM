from abc import ABC, abstractmethod  # abstract base class
import pickle
import base64
import json
import time

import numpy.random
from numpy import linspace, geomspace, power


class AbstractModel(ABC):
    """Abstract base class of PM models."""
    
    def __init__(self, seed=1337):
        numpy.random.seed(seed)
        self._times = {}
    
    @abstractmethod
    def train(self, data_x, data_y):
        """Trains the model using the feature data `data_x` and the corresponding target values `data_y`."""
        pass
    
    @abstractmethod
    def predict(self, data_x):
        """Returns the model's prediction for the feature data `data_x`."""
        pass
    
    @abstractmethod
    def get_model_parameters(self):
        """Collects the model's parameters into a dict."""
        pass
    
    def save_model(self, filename: str, metadata: dict={}):
        """Save the model into a file.
        
        The file is formatted as JSON, optionally including `metadata`.
        Model parameters are always included in the metadata.
        """
        data = {
            "model": base64.b64encode(pickle.dumps(self)).decode('ascii'),  # encodes bytes as text
            "metadata": metadata | {"model_parameters": self.get_model_parameters()} | {"model_type": type(self).__name__}
        }
        with open(filename, "w") as file:
            json.dump(data, file, indent="\t")
    
    @classmethod
    def load_model(cls, filename, return_metadata=False):
        """Load a model from a file.
        
        Returns a tuple `(model, metadata)` if `return_metadata=True`.
        """
        with open(filename, "r") as file:
            data = json.load(file)
        model = pickle.loads(base64.b64decode(data["model"]))
        
        if not return_metadata:
            return model
        else:
            return model, data["metadata"]

    def time_it(self, name, start=True):
        """Times any function 
        """
        if start:
            self._times[name] = time.time()
        else:
            self._times[name] = time.time() - self._times[name]
    
    def _generate_sample_weights(self, n_samples, weighting_strategy, weighting_factor):
        """Returns a weighting for the samples, based on their count and a weighting strategy.
        
        The weighting assumes that the first (oldest) samples should have the lowest weight and that the last (newest) samples should have the highest weight.
        
        The `weighting_strategy` can be "uniform" (no weighting), "linear", or "exponential".
        For "linear" and "exponential", the weights range from 1 to `weighting_factor`.
        
        Additionally, there are the variants "linear_steps" and "exponential_steps", which split the data into equally sized steps,
        where the number of steps is equal to `weighting_factor` and the maximum weight is either `weighting_factor` (linear_steps) or 2^`weighting_factor` (exponential_steps).
        """        
        if weighting_strategy == "uniform":
            # LR, GBM, and NN use None as default value, meaning that no sample weighting is applied
            return None
        if weighting_strategy == "linear":
            return linspace(1, weighting_factor, n_samples)
        if weighting_strategy == "exponential":
            return geomspace(1, weighting_factor, n_samples)
        if weighting_strategy == "linear_steps":
            # weighting_factor is used as number of steps of equal length, and identical to weight at highest step (since the endpoint isn't included)
            return linspace(1, weighting_factor+1, n_samples, endpoint=False, dtype=int)  # rounding down (int) creates steps
        if weighting_strategy == "exponential_steps":
            # similar to linear_steps, but the first step has weight 1 (2^0) and every following step has double the weight of the previous step
            return power(2, linspace(0, weighting_factor, n_samples, endpoint=False, dtype=int))
    
