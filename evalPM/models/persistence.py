from .abstract_model import AbstractModel


class PersistenceModel(AbstractModel):
    """A simple model always repeating the previous target value."""
    
    def __init__(self, feature_to_repeat="pmshift"):        
        self.feature = feature_to_repeat
        
    def train(self, data_x, data_y):
        if self.feature not in data_x:
            raise KeyError("Feature '{}' to repeat by persistence model is missing from feature data.".format(self.feature))
    
    def predict(self, data_x):
        # select the column with the feature to repeat
        return data_x[self.feature].values

    def get_model_parameters(self):
        return {
            "feature_to_repeat": self.feature
        }
