from .features import split_data
from .models.abstract_model import AbstractModel
from .compare import evaluate_prediction, align_observation_and_prediction, ALL_METRICS
from .visual import plot

import pandas as pd
import numpy as np
import itertools
import time
import multiprocessing


def train_and_evaluate_model(model: AbstractModel, training_data, test_data, train_metrics=[], test_metrics=["RMSE"],
                             rde_threshold=15, persist_to_file: str=None, persistence_metadata: dict={}):
    """Trains and evaluates a model.
    
    The `model` is expected to be an instance of a subclass of AbstractModel.
    
    `training_data` and `test_data` need to contain both the feature data and the target values.
    
    Available metrics (for both training and test) are all from compare.ALL_METRICS as well as "time".
    Specify "all" to get all available training or test metrics, respectively.
    Specify `test_metrics`="predictions" and no train_metrics to get the model's predictions on the test data.
    
    `rde_threshold` is used for calculating the RDE metric.
    
    Returns a dict mapping metric names to their values if only test metrics are desired.
    Otherwise, returns a dict mapping the strings "training" and "test" to the individual metrics dicts.
    In the case of `test_metrics`="predictions", returns a dict mapping dates to their predictions.
    
    The model can optionally be persisted into a file given by `persist_to_file`, including `persistence_metadata`.
    """
    if not train_metrics and not test_metrics:
        raise ValueError("At least one metric must be specified.")
    if test_metrics == "predictions" and train_metrics:
        raise ValueError("When predictions on the test data should be returned, no training metrics are supported.")
    
    X_train, y_train = split_data(training_data)
    X_test, y_test = split_data(test_data)
    
    train_start = time.time()
    model.train(X_train, y_train)
    train_time = time.time() - train_start
    
    if train_metrics:
        # usually only test_metrics are relevant, so train predictions and metrics are only calculated on demand
        train_prediction = model.predict(X_train)
        train_results = evaluate_prediction(y_train, train_prediction, rde_threshold)
        train_results["time"] = train_time 
    
    test_start = time.time()
    test_prediction = model.predict(X_test)
    test_time = time.time() - test_start
    
    test_results = evaluate_prediction(y_test, test_prediction, rde_threshold)
    test_results["time"] = test_time 
    
    if persist_to_file:
        model.save_model(persist_to_file, persistence_metadata | {"test_evaluation": test_results})
    
    if test_metrics == "predictions":
        return dict(zip(test_data["date"], test_prediction))
    
    # extract wanted metrics; dicts will be empty if metrics are an empty list
    train_dict = train_results if train_metrics == "all" else dict([(metric, train_results[metric]) for metric in train_metrics])
    test_dict = test_results if test_metrics == "all" else dict([(metric, test_results[metric]) for metric in test_metrics])

    if train_metrics:
        return {"training": train_dict, "test": test_dict}
    else:
        return test_dict  # easier handling in the usual case

def train_and_evaluate_model_of_station(model, training_data, test_data, station, train_metrics=[], test_metrics=["RMSE"], rde_threshold=15, persist_to_file=None, persistence_metadata={}):
    """Wrapper of `train_and_evaluate_model` which extracts a `station`'s data out of the training and test data of multiple stations."""
    return train_and_evaluate_model(model, training_data[station], test_data[station], train_metrics, test_metrics, rde_threshold, persist_to_file, persistence_metadata | {"station": station})

def train_and_evaluate_model_at_all_stations(model: type, training_data, test_data, 
                                             model_parameters: dict[str, dict]={}, 
                                             train_metrics=[], test_metrics=["RMSE"], rde_threshold=15,
                                             persist_to_file: str=None, persistence_metadata: dict={}):
    """Wrapper of `train_and_evaluate_model` which trains and evaluates a separate model per station, for every station of the dataset.
    
    Here, `model` is a type subclassing AbstractModel rather than an instance, such that a new model instance can be created for every station.
    The `model_parameters` facilitate the initialization of each instance using constructor arguments specified as a dict mapping parameter names to their values.
    
    Returns a Dataframe of the metrics or predictions per station.
    
    If the models get persisted, their filenames are appended by the station name.
    """
    stations = training_data.keys()
    station_results = []
    
    for station in stations:
        model_instance = model(**model_parameters)
        station_result = train_and_evaluate_model_of_station(model_instance, training_data, test_data, station, train_metrics, test_metrics, rde_threshold, 
                                                             "{}_{}".format(persist_to_file, station) if persist_to_file else None, persistence_metadata)
        station_results.append(_dict_to_dataframe(station_result, train_metrics, test_metrics, row_label=station))
    
    result_df = pd.concat(station_results, axis=0)
    result_df.index.name = "station"
    
    if test_metrics == "predictions":
        # sort columns for the case of different missing dates per station
        result_df = result_df.sort_index(axis=1)
    
    return result_df

def train_and_evaluate_multiple_models_at_all_stations(models: dict[str, type], training_data, test_data, 
                                                       model_parameters: dict[str, dict]={}, 
                                                       train_metrics=[], test_metrics=["RMSE"], rde_threshold=15, print_progress=True):
    """Wrapper of `train_and_evaluate_model_at_all_stations` which trains and evaluates multiple models per station.
    
    `models` is a dict mapping model names to types subclassing AbstractModel, such that new model instances can be created for every station.
    
    The `model_parameters` facilitate the initialization of each instance using constructor arguments. 
    They are specified as one dict per model name, inside of a dict mapping the model names to these parameter dicts.
    
    Returns a Dataframe of the metrics or predictions per model name and station.
    """
    model_results = {}

    for model_name in models.keys():
        model_results[model_name] = train_and_evaluate_model_at_all_stations(models[model_name], training_data, test_data, 
                                                                             model_parameters.get(model_name, {}), 
                                                                             train_metrics, test_metrics, rde_threshold)
        if print_progress:
            print("finished model {}".format(model_name))
    
    return pd.concat(model_results, axis=0, names=["model", "station"])  # names of MultiIndex levels


def _grid_search_parallel_func(args):
    """Executes the training and evaluation of models for a single parameter combination.
    
    Returns a tuple (`args["var_params"]`, evaluation_result)
    """
    var_params = dict(zip(args["var_param_names"], args["var_params"]))
    model_params = var_params | args["static_params"]  # union of dicts
    
    if args["data_per_station"]:
        param_result = train_and_evaluate_model_at_all_stations(args["model"], args["train_data"], args["test_data"], model_params, args["train_metrics"], args["test_metrics"], args["rde_threshold"])
    else:
        result_dict = train_and_evaluate_model(args["model"](**model_params), args["train_data"], args["test_data"], args["train_metrics"], args["test_metrics"], args["rde_threshold"])
        param_result = _dict_to_dataframe(result_dict, args["train_metrics"], args["test_metrics"])
        
    if args["print_progress"]:
        print("finished parameter combination {}".format(var_params))
        
    return args["var_params"], param_result

def hyperparameter_grid_search(model: type, training_data, test_data, data_per_station=True, 
                               varying_model_parameters: dict[str, list]={}, static_model_parameters: dict[str, any]={},
                               train_metrics=[], test_metrics=["RMSE"], rde_threshold=15, print_progress=True,
                               num_jobs=1, max_tasks_per_child=50, chunk_size=1):
    """Executes a complete grid search across parameter combinations.
    Can thus be used for hyperparameter optimization.
    
    Here, `model` is a type subclassing AbstractModel rather than an instance, such that a new model instance can be created for every parameter combination and station.
    If only a single station is relevant, you can pass that station's training and test data and set `data_per_station` to False.
    
    `varying_model_parameters` is a dict mapping the parameter names to lists of the parameter values that should be explored.
    `static_model_parameters` is a dict mapping parameter names to a single, constant value.
    
    `train_metrics` and `test_metrics` can be used in the same way as in `train_and_evaluate_model`.
    
    The degree of parallelization can be controlled using the parameter `num_jobs`.
    `max_tasks_per_child` determines how often the resources of each job are reset, in order to prevent resource leaks.
    
    Returns a Dataframe of the metrics or predictions per parameter combination and station.
    """
    if not varying_model_parameters:
        raise ValueError("At least one varying model parameter must be specified.")
    if [param for param in varying_model_parameters if param in static_model_parameters]:
        raise ValueError("Varying and static model parameters must not overlap.")
    
    var_param_names, var_param_values = zip(*varying_model_parameters.items())  # zipping items instead of using keys() and values() directly in order to enforce corresponding order of names and values
    results_per_param_combination = {}
    
    def argument_generator():
        # generates arguments to be supplied to the _grid_search_parallel_func
        static_args = {
                "model": model,
                "train_data": training_data,
                "test_data": test_data,
                "var_param_names": var_param_names,
                "static_params": static_model_parameters,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "rde_threshold": rde_threshold,
                "data_per_station": data_per_station,
                "print_progress": print_progress
            }
        for param_combination in itertools.product(*var_param_values):
            yield static_args | {"var_params": param_combination}  # union of dicts

    with multiprocessing.Pool(processes=num_jobs, maxtasksperchild=max_tasks_per_child) as pool:
        # imap rather than map to avoid duplicating the training and test data for every parameter combination all at once; 
        # instead, it lazily creates new copies only when they are needed
        result_tuples = pool.imap(_grid_search_parallel_func, argument_generator(), chunk_size)
        # result_tuples needs to be consumed while the pool is active, because it is produced lazily
        results_per_param_combination = dict(result_tuples)
    
    result_df = pd.concat(results_per_param_combination, axis=0, names=var_param_names)  # names of additional outer MultiIndex levels
    
    if data_per_station:
        return result_df
    else:
        # innermost index level was created by _dict_to_dataframe (a DataFrame needs to have an index), but carries no information
        return result_df.droplevel(-1, axis=0) 


def _create_metrics_columns(train_metrics, test_metrics):
    """Creates an index matching the desired training and test metrics."""
    if train_metrics == "all":
        train_metrics = ALL_METRICS + ["time"]
    if test_metrics == "all":
        test_metrics = ALL_METRICS + ["time"]
    
    if train_metrics:  # if train_metrics is not an empty list
        return pd.MultiIndex.from_arrays([["training"]*len(train_metrics)+["test"]*len(test_metrics), train_metrics+test_metrics])
    else:
        return pd.Index(test_metrics)

def _dict_to_dataframe(results: dict, train_metrics=None, test_metrics=None, row_label=0):
    """Turns a result dict into a Dataframe.
    
    Training and test metrics can be inferred, but stating them explicitly is safer.
    """
    if train_metrics is None:
        train_metrics = list(results.get("training", {}).keys())
    if test_metrics is None:
        test_metrics = list(results.get("test", {}).keys()) if "test" in results else list(results.keys())
        # by definition of the function train_and_evaluate_model, either both of "training" and "test" are keys in the result dict, or neither of them (meaning that the keys are test metrics)
    
    if test_metrics == "predictions":
        # dates (dict keys) become column labels; make sure that predictions are sorted by date
        return pd.DataFrame(results, index=[row_label]).sort_index(axis=1)
    
    df = pd.DataFrame(columns=_create_metrics_columns(train_metrics, test_metrics))
    
    if train_metrics:  # if train_metrics is not empty
        # don't iterate over train_metrics or test_metrics, as these could be the string "all"!
        for metric, value in results["training"].items():
            df.loc[row_label, ("training", metric)] = value
        for metric, value in results["test"].items():
            df.loc[row_label, ("test", metric)] = value
    else:
        # only test metrics relevant, and with a different result structure and column index
        for metric, value in results.items():
            df.loc[row_label, metric] = value
    
    return df

def prepend_index_level(df, level_name, level_value, axis=1):
    # adapted from https://stackoverflow.com/a/56278736
    
    if axis == 0 or axis == "index":
        idx = df.index.to_frame()
    else:  # column index
        idx = df.columns.to_frame()
        
    level_values = [level_value]*len(idx)
    idx.insert(0, level_name, level_values)
    
    if axis == 0 or axis == "index":
        df.index = pd.MultiIndex.from_frame(idx)
    else:  # column index
        df.columns = pd.MultiIndex.from_frame(idx)
    
    return df


def get_model_scores(model: type, training_data, test_data, model_parameters={}, error_type="squared_errors", rde_threshold=15):
    """Trains a separate model per station and calculates the model's scores per station, 
    which can be used for significance testing in the `compare` module.
    
    Here, `model` is a type subclassing AbstractModel rather than an instance, such that a new model instance can be created for every station.
    The `model_parameters` facilitate the initialization of each instance using constructor arguments specified as a dict mapping parameter names to their values.
    
    The `error_type` can be a metric, "squared_errors", or "absolute_errors".
    While metrics are aggregated per station, the latter options are calculated for every individual prediction.
    If `error_type == "RDE"`, adjust the threshold using `rde_threshold`.
    
    Scores are inverted if the `error_type` indicates better model performance through lower values.
    
    Returns a one-dimensional array of all scores across the dataset.
    """
    errors = []
    
    for station in training_data.keys():
        X_train, y_train = split_data(training_data[station])
        X_test, y_test = split_data(test_data[station])
        
        model_instance = model(**model_parameters)
        model_instance.train(X_train, y_train)
        prediction = model_instance.predict(X_test)
        
        if error_type in ALL_METRICS:
            # comparison based on an error metric, aggregated per station
            result = evaluate_prediction(y_test, prediction, rde_threshold)[error_type]
        else:
            # comparison based on errors of individual predictions
            raw_errors = y_test - prediction
            if error_type == "squared_errors":
                result = np.square(raw_errors).values  # .values to return an array instead of a pandas series
            elif error_type == "absolute_errors":
                result = np.abs(raw_errors).values
            else:
                raise ValueError("Unsupported error_type: {}".format(error_type))
        
        # append the station's result to the overall list
        errors.append(result)
    
    # turn the errors into a 1D array for the case of a list per station, i.e. with an error per individual prediction
    scores = np.hstack(errors)
    
    if error_type != "R^2":
        # invert scores for all error types where a smaller number means better performance, since deepsig expects higher numbers to be better
        scores = scores * -1
    
    return scores


def plot_model_of_station(model, training_data, test_data, station):
    """Trains the `model` for the `station` and plots both the observations and the model's predictions."""
    model.train(*split_data(training_data[station]))
    prediction = model.predict(split_data(test_data[station])[0])
    
    combined_data = align_observation_and_prediction(test_data[station], prediction)
    plot(combined_data, title=station)
