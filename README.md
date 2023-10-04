# evalPM

This repository contains the framework `evalPM` for creating, evaluating, and comparing immission models of particulate matter (PM), as well as corresponding example datasets.

The datasets have been created using data from _Sächsisches Landesamt für Umwelt, Landwirtschaft und Geologie_, from _Luftgütemessnetz Brandenburg_, from _Deutscher Wetterdienst_, from _ARPA Lombardia_, from the _European Environment Agency_, from _The Norwegian Meteorological Institute_, and from the _Copernicus Atmosphere Monitoring Service_ and the _Copernicus Climate Change Service_.
The individual datasets and their data sources are documented by a `README.txt` in the corresponding data folders.
Note that all times in the supplied datasets are in UTC. If you use data of locations which are in a different time zone, i.e. especially outside of Europe, it is advised to use timestamps of the corresponding local time, since the feature calculation mainly aggregates by date (as specified in the datasets), which obviously should align with the actual local days.

## Using the Framework

A simple example of using the framework may look as follows:

```python
import evalPM as pm
training_data, test_data = pm.features.load_data([2020], [2021])
pm.helpers.train_and_evaluate_model_at_all_stations(pm.models.GradientBoostingRegressor,
                                                    training_data,
                                                    test_data,
                                                    test_metrics=["RMSE", "SMAPE"])
```

Here, a separate Gradient Boosting model is trained per station on data from 2020 and tested on data of the same station from 2021, with the evaluation metrics RMSE and SMAPE being reported per station.

Notable options of `load_data()` are especially the lists of training and test years, but it also supports the argument `feature_file` which should point to a JSON file specifying the features to calculate on the raw data (the format of such files is defined below in the section [Feature Specification](#feature-specification)).
Additionally, it supports the argument `dirpath` which should point to a directory containing the raw data files (one per station), on which the features are to be calculated; the default is `data/data-saxony/`.

The module `pm.helpers` contains many functions for easy model training and evaluation.
As the basis, the function `train_and_evaluate_model()` expects a model instance and DataFrames with training and test data. The model needs to be an instance of any class in the module `pm.models`; if you would like to add a different kind of model, just extend the class `AbstractModel` to ensure compatibility with the framework.
Additional parameters of `train_and_evaluate_model()` are `train_metrics` and `test_metrics`, which both can be either a list of error metrics (see the example above) or the string `"all"` to return all available metrics. `test_metrics` may also be set to `"predictions"` to return the model's predictions rather than error metrics.

More advanced functions of the `pm.helpers` module include `train_and_evaluate_model_at_all_stations()` which expects a model type rather than a model instance and dictionaries mapping station names to their training and test data DataFrames (as returned by `load_data()`), while also supporting `model_parameters` with which to instantiate the model of each station as well as the same `train_metrics` and `test_metrics` as `train_and_evaluate_model()`.
In order to not only train and evaluate a separate model per station, but also vary the model parameters, you can use the function `hyperparameter_grid_search()`; it takes mainly the same arguments as `train_and_evaluate_model_at_all_stations()`, but differentiates between `static_model_parameters` and `varying_model_parameters`, across which all parameter combinations are used to train and evaluate separate models.
Additionally, it can train and evaluate multiple models in parallel, if supplied with the according `num_jobs` argument; note however that using this function, because of the potentially parallel execution, might require you to call it from within a `if __name__ == "__main__":` block.

Trained models can also be persisted in order to load them later on.
This can be done for an individual model or using the helper functions `train_and_evaluate_model()` and `train_and_evaluate_model_at_all_stations()`.
It is advisable to save relevant metadata alongside the model (feature set, dataset, model type ...), which can be done using the parameter `persistence_metadata`.
To load a saved model, use the method `load_model()` of the appropriate model class.

For further details regarding available functions and their arguments, take a look at the code and the docstrings within it.

### Dependencies

This framework uses the following libraries:

- `pandas` (version 1.4)
- `numpy` (version 1.22)
- `scikit-learn` (version 1.0)
- `tensorflow` (version 2.8)
- `deepsig` (version 1.2)
- `matplotlib` (version 3.5)

### experiment.py

To run an extensive grid search across model types, the amount of training data, boundary layer height features, and hyperparameters, you can simply run the script `experiment.py` which utilizes this framework.
Note that the runtime of the unmodified script will be multiple days.
To get the same results as us, make sure to install the same dependency versions using `pip` and the supplied `requirements.txt`.
The script has been tested with python version `3.9.12`.
Result files will be written to the directory `results`.
To switch to a different dataset than Saxony, simply adjust the variables `data_folder`, `training_years`, and `test_years` inside the script as needed.

## Feature Specification

Features are specified by a JSON file, which can be passed as an argument to the `load_data` function. The default file is `features_default.json`, which is based on the features of [Klingner et al.](https://doi.org/10.1127/0941-2948/2008/0288) (the files `features_blh-mean.json` and `features_blh-min-max.json` are based on them as well).

The structure of the JSON file has to be as follows:

```json
{
    "date_attribute": "date",
    "time_attribute": "time",
    "features": [
        {}
    ]
}
```

`date_attribute` and `time_attribute` specify the names of the date and time column in the data files.

`features` is a comma-separated list of feature objects, each of which has the following structure:

```json
{
    "name": "rain2",
    "source_attribute": "rain",
    "aggregation": {
        "type": "mean",
        "start": "12:30:00",
        "end": "16:00:00"
    },
    "shift": 0,
    "delta": 0
}
```

All values from `start` to `end` of each day (both times are inclusive; corresponding to the `time_attribute`) of the column specified as `source_attribute` get grouped by the `date_attribute` and aggregated according to the aggregation `type`. The name of the resulting feature is specified by `name`. In this case, the mean of all rain values from 12:30 to 16:00 of each day would become the feature `rain2`.

Possible values of the aggregation `type` include `sum, mean, median, min, max, prod, std, var`.

If the `start` time is after the `end` time or both are equal, the time range is interpreted to be from one day to the next day, e.g. from 16:30 to 06:30. In that case, the date of the calculated feature is the date of the latter day (i.e. the time range is from the previous day to the current day).

To aggregate over the whole day, simply omit both `start` and `end`.

The `shift` parameter can be used to shift the calculated feature by n days, e.g. to get an average value of the previous day (`shift=1`). The `delta` parameter can be used to calculate differences of the feature between n days, e.g. the change of the feature from the previous to the current day (`delta=1`). Both `shift` and `delta` can also be omitted instead of setting them to `0`.

### Special cases

In addition to these general options, there are also a few special types of aggregation.

Firstly, there is the aggregation type `delta`, which calculates a difference between two values of the `source_attribute` per day. The timestamps of these values are specified via `start` and `end` (i.e. they are required for this aggregation type); specifying a `start` greater than or equal to `end` results in `start` being on the previous day here as well. The feature is calculated as the end value minus the start value. This is especially useful for calculating the change of a variable within one day:

```json
{
    "name": "temp_gradient",
    "source_attribute": "temperature",
    "aggregation": {
        "type": "delta",
        "start": "04:00:00",
        "end": "12:00:00"
    }
}
```

Secondly, there are the aggregation types `consecutive_days_zero` and `consecutive_days_nonzero`, which count the number of consecutive preceding days (including the current day) where the `source_attribute` is zero throughout the whole day (`consecutive_days_zero`) or nonzero for at least one timestamp of the day (`consecutive_days_nonzero`), respectively.
It is optionally possible to specify a time range via `start` and `end` in order to only consider the specified part of each day.
These aggregation types could e.g. be used to calculate the number of preceding days with or without precipitation:

```json
{
    "name": "days_without_rain",
    "source_attribute": "rain",
    "aggregation": {
        "type": "consecutive_days_zero"
    }
}
```

Thirdly, there is the option to group by a different column than the `date_attribute`. To achieve this, set the `aggregation` block's optional parameter `group_by` to the column name by which to group the `source_attribute`. The necessary mapping from the resulting group keys to the `date_attribute` groups is found by grouping the `group_by` column by the `date_attribute` (each day should only have a single unique value (i.e. repeating for every time of that day) of the `group_by` column, so the mean is always used for this aggregation). Do NOT use this parameter to group by the `date_attribute`; for that case, simply omit the `group_by` parameter, since grouping by the `date_attribute` is the default behavior. Grouping by a different column than the `date_attribute` currently does not support selecting a sub-range of the `time_attribute` via `start` and `end`. An example use-case for this aggregation option is calculating the mean PM concentration per weekday, and supplying this value as a feature to every date with that weekday:

```json
{
    "name": "pm_weekday",
    "source_attribute": "pm",
    "aggregation": {
        "type": "mean",
        "group_by": "weekday"
    }
}
```

NOTE: The aggregation types `delta` and `consecutive_days_(non)zero` are mutually exclusive to specifying a different `group_by` column, i.e. they cannot be performed within a single feature. However, all of these can optionally be combined with either or both of `shift` and `delta` (outside of the `aggregation` block, see the general example above).

## Implementation

The framework consists of five modules, which interact with each other, as well as datasets.

The datasets consist of one CSV file per station, which contain columns for the date, time, and measurements.
These datasets are used by the `features` module to calculate various features as well as the target variable using a feature definition as detailed above.

The `models` module acts as an interface between the framework and specific model implementations from other libraries.
Therefore, existing implementations of various model types can be used and are available through a uniform interface within the framework.
All hyperparameters are passed to the constructor of the relevant interface class, such that the training and prediction methods only require the actual training or test data.
The currently included model interfaces include _linear regression_ and _gradient boosting_ based on `scikit-learn` as well as _feedforward neural networks_ based on `tensorflow`.
However, more interfaces can be simply added as needed, as long as they adhere to the definition of the `AbstractModel`.

The `compare` module facilitates evaluation and comparison of model predictions, either through various metrics like RMSE and SMAPE or through a statistical test based on `deep-significance` (model comparison only).

The `visual` module offers functions for visual interpretations by using `matplotlib`, either through a line plot of observations and predictions or through a heatmap of evaluation results.

The `helpers` module utilizes the modules `models` and `compare` to wrap model training and evaluation into a single function call and thereby simplify the use of the framework.
This module also includes a function for hyperparameter optimization using grid search.

## Attribution

Material preparation, data collection, analysis, and implementation were performed by [Jonas Deepe](https://github.com/JonasDeepe) and [Lucas Woltmann](https://github.com/lucaswo).
