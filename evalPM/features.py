import os
import json
import pandas as pd


def _select_time_range(data, aggregation_spec, date_var, time_var):
    """Trims the `data` to the time range specified in the `aggregation_spec` (inclusive at both start and end).
    
    If no time range is specified, the original data is returned.
    
    If the start time is after the end time, or the start time equals the end time, 
    the time range is interpreted to cross from one day to the next day, e.g. from 18:00 to 06:00.
    """
    if "start" not in aggregation_spec:
        return data
        
    start = pd.to_datetime(aggregation_spec["start"]).time()
    end = pd.to_datetime(aggregation_spec["end"]).time()
    if start < end:
        return data[(data[time_var] >= start) & (data[time_var] <= end)]
    else:  
        # start after end, or start == end
        prev_day = data[data[time_var] >= start].copy()
        next_day = data[data[time_var] <= end].copy()
        prev_day[date_var] = prev_day[date_var].add(pd.Timedelta(days=1))  # shifting the dates by one day to align corresponding values on the same date_var
        return pd.concat([prev_day, next_day])

def _calculate_features(data: pd.DataFrame, feature_spec: dict):
    """Calculates the features specified in `feature_spec` from the `data`.
    
    Feature specification is documented in the top-level README file.
    
    Every row missing at least one feature is dropped to ensure complete data.
    Returns a Dataframe sorted by date.
    """
    date_var = feature_spec["date_attribute"]
    time_var = feature_spec["time_attribute"]
    features = feature_spec["features"]
    
    # converting dates and times to datetime objects for proper handling
    data[date_var] = pd.to_datetime(data[date_var], infer_datetime_format=True).dt.date
    data[time_var] = pd.to_datetime(data[time_var], infer_datetime_format=True).dt.time
    
    # initial column with unique dates
    feature_data = pd.DataFrame(data[date_var].unique(), columns=[date_var])  # column name must be manually set because unique() returns an ndarray
    
    for feature in features:
        feature_name = feature["name"]
        feature_source = feature["source_attribute"]
        aggregation_spec = feature["aggregation"]
        
        if aggregation_spec["type"] == "delta":
            # calculating a difference between two time_var values per date_var
            
            if "group_by" in aggregation_spec:
                raise ValueError("Aggregation type 'delta' does not support grouping by a different attribute than the date_attribute.")
            
            start = pd.to_datetime(aggregation_spec["start"]).time()
            end = pd.to_datetime(aggregation_spec["end"]).time()
            
            start_values = data.loc[data[time_var] == start, [date_var, feature_source]].copy()
            end_values = data.loc[data[time_var] == end, [date_var, feature_source]].copy()
            if start >= end:
                # if start >= end, start is interpreted to be on the previous day, and thus the start_values need to be shifted by one day to be aligned with the corresponding end_values
                start_values[date_var] = start_values[date_var].add(pd.Timedelta(days=1))
            
            # subtraction is index-based, i.e. it is applied on rows with identical index (date_var) value
            start_values = start_values.set_index(date_var)
            end_values = end_values.set_index(date_var)
            aggregated_feature = end_values.sub(start_values)
            
        elif aggregation_spec["type"] in ["consecutive_days_zero", "consecutive_days_nonzero"]:
            # counting the number of consecutive preceding days (including the current day) where the source_attribute is zero for the whole day (or not)
            
            if "group_by" in aggregation_spec:
                raise ValueError("Aggregation type {} does not support grouping by a different attribute than the date_attribute.".format(aggregation_spec["type"]))
            
            selected_data = _select_time_range(data, aggregation_spec, date_var, time_var)
            
            daily_data = selected_data.groupby(date_var).agg({feature_source: "any"})  # days get "True" if any value on that day is True (or nonzero)
            if aggregation_spec["type"] == "consecutive_days_zero":
                # invert boolean Series if number of consecutive original zeroes is relevant
                daily_data = ~daily_data
            
            # sorting the data for the upcoming cumulative sums
            daily_data = daily_data.sort_index()
            
            # adapted from https://stackoverflow.com/a/73940626; start of consecutive identical values is identified, consecutive values are grouped, and their boolean values cumulatively summed
            aggregated_feature = daily_data.groupby(daily_data.ne(daily_data.shift()).cumsum()[feature_source]).agg({feature_source: "cumsum"})
            
        elif "group_by" in aggregation_spec:
            # grouping by a different attribute than date_var; required for "weekday" attribute
            
            if "start" in aggregation_spec:
                raise ValueError("Grouping by a different attribute than the date_attribute currently does not support selecting a sub-range of the time_attribute.")
            
            grouping_attribute = aggregation_spec["group_by"]
            
            # first, create a mapping from date_var to the grouping_attribute; renaming the column to avoid potential conflicts when merging:
            target_grouping = data.groupby(date_var).agg({grouping_attribute: "mean"}).rename(columns={grouping_attribute: "grouping_attribute"})
            # then, calculate the wanted feature by grouping by the grouping_attribute (which becomes the index):
            grouped_feature = data.groupby(grouping_attribute).agg({feature_source: aggregation_spec["type"]})
            # finally, merge the feature to date_var using the grouping_attribute; after merging, it can be dropped:
            aggregated_feature = target_grouping.merge(grouped_feature, left_on="grouping_attribute", right_index=True, how="left").drop(columns="grouping_attribute")
            
        else:
            # regular aggregation based on date_var, potentially using a limited range of time_var
            selected_data = _select_time_range(data, aggregation_spec, date_var, time_var)
            aggregated_feature = selected_data.groupby(date_var).agg({feature_source: aggregation_spec["type"]})  # date_var becomes index
        
        # renaming the aggregated column into feature_name (date_var is the index)
        aggregated_feature.columns = [feature_name]
        
        
        if "shift" in feature:
            # shifting the data by adding n days to date_var (the index); this doesn't change the feature data, so it can safely be applied before a potentially following "delta"
            aggregated_feature.index = pd.Series(aggregated_feature.index).add(pd.Timedelta(days=feature["shift"]))
            
        if "delta" in feature and feature["delta"] != 0:
            # shifting the data by n days, and then subtracting the shifted version from the un-shifted version (e.g. current day minus (shifted) previous day, i.e. the change from previous to current day)
            shifted_feature = aggregated_feature.set_index(pd.Series(aggregated_feature.index).add(pd.Timedelta(days=feature["delta"])))
            aggregated_feature = aggregated_feature.sub(shifted_feature)  # subtraction is index-based, i.e. it is applied on rows with identical index (date_var) value
            
        
        # merging the aggregated feature onto the already calculated features;
        # using a right outer join in case a key is not among the initial set of date_var, but added by every feature (using shift); 
        # a key not present in the current feature would get dropped anyway by the finalizing dropna(), meaning that a full outer join is not necessary
        feature_data = feature_data.merge(aggregated_feature, left_on=date_var, right_index=True, how="right")
    
    return feature_data.dropna().sort_values(by=date_var, ignore_index=True)


def _load_raw_station_data(dirpath, station, train_years, test_years, additional_paths=[], date_time_vars=["date", "time"]):
    """Loads the data of a single station.
    
    The data is expected to be a CSV file named `station`.csv, located at `dirpath`.
    
    If `additional_paths` are specified, they are expected to contain a `station`.csv file as well,
    and their data is merged using the `date_time_vars`.
    
    Loaded data is restricted to `train_years` and `test_years`, respectively.
    Returns a tuple `(raw_training_data, raw_test_data)`.
    """
    raw_station_data = pd.read_csv("{}{}.csv".format(dirpath, station))
    
    for path in additional_paths:
        additional_dataset = pd.read_csv("{}{}.csv".format(path, station))
        raw_station_data = raw_station_data.merge(additional_dataset, how="left", on=date_time_vars)
    
    raw_station_data.sort_values(by=date_time_vars, ignore_index=True)
    
    raw_train_data = raw_station_data.loc[pd.to_datetime(raw_station_data[date_time_vars[0]]).dt.year.isin(train_years)]
    raw_test_data = raw_station_data.loc[pd.to_datetime(raw_station_data[date_time_vars[0]]).dt.year.isin(test_years)]
    
    return raw_train_data, raw_test_data

def _check_loaded_data(data, date_var, years, station):
    """Checks that every desired year is contained in `data`.
    Prints warnings if that is not the case, or if a particular year contains only few data points.
    """
    rows_per_year = data.groupby(pd.to_datetime(data[date_var]).dt.year).size()
    for year in years:
        if year not in rows_per_year.index:
            print("Station {} has no data for {}!".format(station, year))
        elif rows_per_year.loc[year] < 300 * 24:
            print("Station {} might have insufficient data for {}! ({} total timestamps)".format(station, year, rows_per_year.loc[year]))

def load_data(training_years=[2018], test_years=[2019], dirpath="data/data-saxony/", 
              additional_dirpaths=["data/ERA5-features/"], exclude_stations=[],
              feature_file="features_default.json", feature_engine=_calculate_features, 
              print_result=True, print_warnings=True):
    """Loads training and test data for all stations of a dataset, and calculates the desired features for the specified years.
    
    A dataset can be selected using the `dirpath` parameter.
    Additional data for the same set of stations can be selected using `additional_dirpaths`.
    If not all stations should be loaded, individual stations can be excluded using `exclude_stations`.
    If no training or test data is found for one of the stations, that stations is skipped.
    
    The `feature_file` needs to define the desired features to calculate on the raw data.
    The details of this feature specification are documented in the top-level README file.
    
    Returns a tuple `(training_data, test_data)`.
    """
    training_data = {}
    test_data = {}

    stations = [entry.name.rsplit(".", 1)[0] for entry in os.scandir(dirpath) if entry.is_file()]
    if "README" in stations:
        # remove readme file from station list in case it exists
        stations.remove("README") 
    for unwanted_station in exclude_stations:
        # remove unwanted stations from the station list
        if unwanted_station in stations:
            stations.remove(unwanted_station)
    
    # parse feature definitions and extract the names of date and time columns
    with open(feature_file) as file:
        feature_spec = json.load(file)
    date_var = feature_spec["date_attribute"]
    time_var = feature_spec["time_attribute"]
    date_time_vars = [date_var, time_var]
    
    for station in stations:
        raw_training_data, raw_test_data = _load_raw_station_data(dirpath, station, training_years, test_years, 
                                                                  additional_dirpaths, date_time_vars)
        
        if print_warnings:
            _check_loaded_data(raw_training_data, date_var, training_years, station)
            _check_loaded_data(raw_test_data, date_var, test_years, station)
        
        if raw_training_data.empty or raw_test_data.empty:
            if print_warnings:
                print("Skipping station {} because it has no relevant training or test data!".format(station))
            continue
        
        training_data[station] = feature_engine(raw_training_data, feature_spec)
        test_data[station] = feature_engine(raw_test_data, feature_spec)
    
    if print_result:
        print("loaded training and test data of {} stations".format(len(training_data)))  # training and test data cannot have a different number of stations
    
    return training_data, test_data


def split_data(combined_data: pd.DataFrame, target="pm", columns_to_ignore=["date"]):
    """Splits a DataFrame into input features (X) and `target` variable (y).
    
    Columns which should not be included in either X or y can be specified using `columns_to_ignore` ("date" by default).
    
    Returns a tuple `(X, y)`.
    """
    features = [feature for feature in combined_data.columns if (feature not in columns_to_ignore and feature != target)]
    X_data = combined_data.loc[:, features]
    y_data = combined_data.loc[:, target]
    return X_data, y_data

