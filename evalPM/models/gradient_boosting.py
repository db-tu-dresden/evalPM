from .abstract_model import AbstractModel

from sklearn.ensemble import GradientBoostingRegressor as GBM


class GradientBoostingRegressor(AbstractModel):
    """A wrapper of sklearn.ensemble.GradientBoostingRegressor."""
    
    def __init__(self, n_trees=75, max_depth=2, min_samples_per_leaf=20, seed=1337, 
                 sample_weight_strategy="uniform", sample_weight_factor=1):
        super().__init__(seed)
        
        self.model = GBM(n_estimators=n_trees, max_depth=max_depth, min_samples_leaf=min_samples_per_leaf, random_state=seed)
        
        self.weighting_strategy = sample_weight_strategy
        self.weighting_factor = sample_weight_factor
        
        self._hyperparameters = {
            "n_trees": n_trees, 
            "max_depth": max_depth, 
            "min_samples_per_leaf": min_samples_per_leaf, 
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
