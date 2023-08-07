import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # don't print tensorflow INFO and WARNING messages

import evalPM as pm
import pandas as pd

model_types = ["LR", "GBM", "FNN"]
model_classes = {
    "LR": pm.models.LinearRegressor,
    "GBM": pm.models.GradientBoostingRegressor,
    "FNN": pm.models.FeedforwardNeuralNetwork
}

data_folder = "data/data-saxony/"

training_years = list(range(2010,2021))
test_years = [2021]
training_year_counts = range(1, len(training_years) + 1)  # from one to all potential training years

blh_feature_options = ["none", "mean", "min/max"]
feature_files = {
    "none": "features_default.json",
    "mean": "features_blh-mean.json",
    "min/max": "features_blh-min-max.json"
}
feature_counts = {  # depending on blh_feature_option
    "none": 12,
    "mean": 13,
    "min/max": 14
}

weighting_strategies = ["uniform", "exponential_steps", "linear_steps"]

lr_parameters = {
    "sample_weight_strategy": weighting_strategies
}

gbm_parameters = {
    "n_trees": list(range(5, 101, 5)),
    "max_depth": list(range(1, 8)),
    "min_samples_per_leaf": [1] + list(range(5, 101, 5)),
    "sample_weight_strategy": weighting_strategies
}

fnn_parameters = {
    "width": list(range(4, 17, 2)),
    "depth": list(range(1, 5)),
    "activation_function": ["default", "linear", "relu"],
    "dropout_rate": [0, 0.3, 0.5],
    "batch_size": [16, 32],
    "sample_weight_strategy": weighting_strategies
}

varying_parameters = {
    "LR": lr_parameters,
    "GBM": gbm_parameters,
    "FNN": fnn_parameters
}

output_dir = "results/"


if __name__ == "__main__":

    # extensive grid search of model type, amount of training data, blh features, and hyperparameters

    for model in model_types:
        model_results_per_year_count = {}
        
        for number_training_years in training_year_counts:
            year_count_results_per_feature_set = {}
            
            static_parameters = {"sample_weight_factor": number_training_years}
            
            for blh_features in blh_feature_options:
                training_data, test_data = pm.features.load_data(training_years[-number_training_years:], test_years, 
                                                                dirpath=data_folder, feature_file=feature_files[blh_features],
                                                                print_result=False)
                
                if model == "FNN":
                    static_parameters["num_features"] = feature_counts[blh_features]
                
                hyperparameter_result = pm.helpers.hyperparameter_grid_search(model_classes[model], training_data, test_data, 
                                                        varying_model_parameters=varying_parameters[model], 
                                                        static_model_parameters=static_parameters,
                                                        train_metrics="all", test_metrics="all", 
                                                        print_progress=False, num_jobs=4)
                
                year_count_results_per_feature_set[blh_features] = hyperparameter_result
                print("finished model {}: year count {}, blh feature {}".format(model, number_training_years, blh_features))
            
            year_count_results = pd.concat(year_count_results_per_feature_set, axis=0, names=["blh_features"])
            year_count_results.to_csv("{}temp-result-{}.csv".format(output_dir, number_training_years))
            model_results_per_year_count[number_training_years] = year_count_results
        
        model_result = pd.concat(model_results_per_year_count, axis=0, names=["training_year_count"])
        model_result.to_csv("{}{}-results.csv".format(output_dir, model))
        print("finished model {} completely".format(model))
