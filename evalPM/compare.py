import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from deepsig import aso


ALL_METRICS = ["MSE", "RMSE", "MAE", "SMAPE", "R^2", "RDE"]
"""List of all metrics' names which can be calculated using this module."""


def smape(y_true, y_pred):
    """Calculates the symmetric mean absolute percentage error between observations and predictions."""
    return np.nan_to_num(200*(np.abs(y_pred-y_true)/(np.abs(y_true) + np.abs(y_pred)))).mean()  # replacing NaNs with 0, as they are the result of y_true and y_pred both being 0, i.e. 0 error

def relative_directive_error(y_true, y_pred, threshold):
    """Calculates the relative directive error between observations and predictions based on a threshold, as defined by the EEA."""
    y_true_sort = np.sort(y_true)
    y_pred_sort = np.sort(y_pred)
    y_true_threshold = np.abs(y_true_sort - threshold).argmin()
    rel_error = np.abs(y_pred_sort[y_true_threshold]-y_true_sort[y_true_threshold])/threshold*100
    return rel_error



def evaluate_prediction(observation, prediction, threshold=15):
    """Calculates all available metrics to evaluate the prediction against the observation.
    Returns the results as a dictionary.
    """
    mse = mean_squared_error(observation, prediction)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(observation, prediction)
    s_mape = smape(observation, prediction)
    r2 = r2_score(observation, prediction)
    rde = relative_directive_error(observation, prediction, threshold)
    
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "SMAPE": s_mape, "R^2": r2, "RDE": rde}


def evaluate_CAMS_forecast(CAMS_model: str, test_data: dict, date_col="date", observation_col="pm"):
    """Calculates all available metrics to evaluate a CAMS forecast at all stations of the dataset.
    Returns the results as a Dataframe.
    """
    stations = test_data.keys()
    result_df = pd.DataFrame(index=stations, columns=ALL_METRICS)
    
    for station in stations:
        station_result = evaluate_CAMS_forecast_at_station(CAMS_model, station, test_data[station], date_col, observation_col)
        
        for metric, value in station_result.items():
            result_df.loc[station, metric] = value
    
    return result_df

def evaluate_CAMS_forecast_at_station(CAMS_model: str, station: str, station_test_data: pd.DataFrame, date_col="date", observation_col="pm"):
    """Calculates all available metrics to evaluate a CAMS forecast at a given station.
    Returns the results as a dictionary.
    """
    cams_data = pd.read_csv("data/CAMS-forecasts/{}.csv".format(station), index_col=0, parse_dates=True)
    cams_data.index = cams_data.index.date  # turn datetime64 objects into date objects
    
    merged_data = station_test_data[[date_col, observation_col]].merge(cams_data, left_on=date_col, right_index=True)[[observation_col, CAMS_model]].dropna()
    
    if len(merged_data) == 0:
        print("No corresponding CAMS values to evaluate for station {}!".format(station))
        return dict()
    elif len(merged_data) < 300:
        print("Evaluating on only {} values for station {}!".format(len(merged_data), station))
    
    return evaluate_prediction(merged_data[observation_col], merged_data[CAMS_model])


def align_observation_and_prediction(observations, predictions, date_col="date", observation_col="pm"):
    """Aligns observations and predictions into a single Dataframe with the date as index.
    The columns get named `observation` and `prediction`.
    """
    aligned_data = observations[[date_col, observation_col]].set_index(date_col)
    aligned_data.index = pd.to_datetime(aligned_data.index)
    aligned_data.rename(columns={observation_col: "observation"})
    aligned_data.loc[:, "prediction"] = predictions
    return aligned_data


def significance_test(scores_a, scores_b, confidence_level=0.95, iteration_count=1000, num_jobs=4, seed=1337, print_progress=False):
    """Performs a significance test of `scores_a` against `scores_b` using `deep-significance`.
    
    It is assumed that a higher score means better model performance.
    Therefore, error measures like the RMSE need to be multiplied by -1 beforehand.
    """
    min_eps = aso(scores_a, scores_b, confidence_level=confidence_level, num_bootstrap_iterations=iteration_count, 
                  num_jobs=num_jobs, show_progress=print_progress, seed=seed)
    return min_eps

def multi_significance_test(model_scores: dict, confidence_level=0.95, iteration_count=1000, num_jobs=4, seed=1337, print_progress=True):
    """Performs a significance test of each set of scores against each other set of scores using `deep-significance`.
    
    `model_scores` is expected to be a dictionary mapping model names to their scores.
    
    It is assumed that a higher score means better model performance.
    Therefore, error measures like the RMSE need to be multiplied by -1 beforehand.
    """
    result = pd.DataFrame()
    
    for model_a in model_scores.keys():
        for model_b in model_scores.keys():
            if model_a == model_b:
                min_eps = 1.0
            else:
                min_eps = significance_test(model_scores[model_a], model_scores[model_b], confidence_level, iteration_count, num_jobs, seed, False)
            
            result.loc[model_a, model_b] = min_eps
            
            if print_progress:
                print("finished {} vs {}".format(model_a, model_b))
                
    return result
