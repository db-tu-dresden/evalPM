from .abstract_model import AbstractModel

from sklearn.linear_model import LinearRegression as LR


class LinearRegressor(AbstractModel):
    """A wrapper of sklearn.linear_model.LinearRegression."""
    
    def __init__(self, seed=1337, sample_weight_strategy="uniform", sample_weight_factor=1):
        super().__init__(seed)
        
        self.model = LR()
        
        self.weighting_strategy = sample_weight_strategy
        self.weighting_factor = sample_weight_factor
        
        self._hyperparameters = { 
            "seed": seed,
        }
        
    def train(self, data_x, data_y):
        sample_weights = super()._generate_sample_weights(len(data_x), self.weighting_strategy, self.weighting_factor)
        self.model.fit(data_x, data_y, sample_weight=sample_weights)
    
    def predict(self, data_x):
        return self.model.predict(data_x)
    
    def get_model_parameters(self):
        return self._hyperparameters | {
            "sample_weight_strategy": self.weighting_strategy, 
            "sample_weight_factor": self.weighting_factor
        }
