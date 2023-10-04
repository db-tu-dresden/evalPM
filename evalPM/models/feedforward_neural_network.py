from .abstract_model import AbstractModel

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import backend
from tensorflow.random import set_seed

import json


class FeedforwardNeuralNetwork(AbstractModel):
    """A wrapper of a feedforward neural network model based on tensorflow."""
    
    def __init__(self, width=10, depth=1, num_features=12, seed=1337, 
                 loss="mean_squared_error", activation_function="default", dropout_rate=None, 
                 batch_size=16, epochs=200, callbacks=[], 
                 sample_weight_strategy="uniform", sample_weight_factor=1):
        backend.clear_session()
        super().__init__(seed)
        set_seed(seed)
        
        self.model = _build_network(width, depth, num_features, loss, activation_function, dropout_rate)
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.callbacks = callbacks
        
        self.weighting_strategy = sample_weight_strategy
        self.weighting_factor = sample_weight_factor
        
        self._hyperparameters = {
            "width": width, 
            "depth": depth, 
            "num_features": num_features, 
            "seed": seed,
            "loss": loss,
            "activation_function": activation_function,
            "dropout_rate": dropout_rate
        }
        
        
    def train(self, data_x, data_y):
        # early stopping: callbacks = [EarlyStopping(monitor="loss", min_delta=0.05, patience=3)]
        
        self.time_it("training")
        sample_weights = super()._generate_sample_weights(len(data_x), self.weighting_strategy, self.weighting_factor)
        self.model.fit(data_x, data_y, sample_weight=sample_weights, shuffle=True, 
                       batch_size=self.batch_size, epochs=self.epochs, verbose=0, callbacks=self.callbacks)
        self.time_it("training", start=False)
        
    
    def predict(self, data_x):
        self.time_it("predict")
        preds = self.model.predict(data_x)[:,0]
        self.time_it("predict", start=False)
        return preds

    def get_model_parameters(self):
        return self._hyperparameters | {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "sample_weight_strategy": self.weighting_strategy, 
            "sample_weight_factor": self.weighting_factor
        }
    
    def save_model(self, filename: str, metadata: dict={}):
        """Save the model into a file (if `filename` ends in `.keras`) or a directory.
        
        `metadata` are saved into a separate JSON file, including model parameters.
        """
        self.model.save(filename)
        with open("{}_metadata.json".format(filename.rsplit(".", 1)[0]), "w") as file:
            json.dump(metadata | {"model_parameters": self.get_model_parameters()} | {"model_type": type(self).__name__}, file, indent="\t")

    @classmethod
    def load_model(cls, filename, return_metadata=False):
        """Load a model from a file.
        
        Returns a tuple `(model, metadata)` if `return_metadata=True`.
        """
        tf_model = load_model(filename)
        with open("{}_metadata.json".format(filename.rsplit(".", 1)[0]), "r") as file:
            metadata = json.load(file)
        
        model = cls(**metadata["model_parameters"])
        model.model = tf_model
        
        if not return_metadata:
            return model
        else:
            return model, metadata


def _build_network(width, depth, num_features, loss, activation_function, dropout_rate):
    
    first_layer_activation = "linear" if activation_function == "default" else activation_function
    next_layers_activation = "relu" if activation_function == "default" else activation_function
    
    one_input = Input(shape=(num_features,), name='one_input')
    x = Dense(width, activation=first_layer_activation, kernel_initializer='uniform')(one_input)
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)
    
    for i in range(1,depth):
        x = Dense(width, activation=next_layers_activation, kernel_initializer='uniform')(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        
    x = Dense(1, kernel_initializer='uniform', name="main_output", activation="linear")(x)
    
    model = Model(inputs=one_input, outputs=x)
    model.compile(loss=loss, optimizer="adam")
    
    return model
