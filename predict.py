#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate prediction scores for the testing file
"""

from __future__ import print_function

import argparse
import pandas as pd
import numpy as np

from copy import deepcopy
from joblib import load

__author__ = "Eric Khoo Jiun Hooi"
__copyright__ = "Copyright 2019, Grab AI For S.E.A."
__credits__ = ["Eric Khoo Jiun Hooi"]
__email__ = "erichooi1995@gmail.com"
__license__ = "MIT"

class StackingRegressor:
    def __init__(self, models, second_model, features):
        self.models = models
        self.feature_models = []
        self.second_model = second_model
        self.features = features
    
    def _generate_f_features(self, X):
        f_features = np.zeros((X.shape[0], len(self.features) * len(self.models)))
        for num, features in enumerate(self.features * len(self.models)):
            model = self.feature_models[num]
            f_features[:, num] = model.predict(X.loc[:, features[1]])
        return f_features
    
    def fit(self, X, y):
        # generate multiple trained models with different features
        for model in self.models:
            for feature in self.features:
                model.fit(X.loc[:, feature[1]], y)
                self.feature_models.append(deepcopy(model))
        f_features = self._generate_f_features(X)
        self.second_model.fit(f_features, y)
    
    def predict(self, X):
        f_features = self._generate_f_features(X)
        return self.second_model.predict(f_features)

def parser():
    parser = argparse.ArgumentParser(description="Script to predict whether the driving trips are safe or dangerous")
    parser.add_argument("-t", "--test-file", type=str, nargs=1, required=True, help="testing data (csv file)", dest="tf")
    return parser

if __name__ == "__main__":
    parser = parser()
    args = parser.parse_args()

    test_file = args.tf[0]
    test_data = pd.read_csv(test_file)

    test_X = pd.DataFrame()

    for col in test_data.columns:
        if col != "bookingID":
            temp = test_data.groupby("bookingID")[col].agg(["mean", "sum", "max", "min"])
            test_X[col + "_mean"] = temp["mean"]
            test_X[col + "_sum"] = temp["sum"]
            test_X[col + "_max"] = temp["max"]
            test_X[col + "_min"] = temp["min"]
    
    bookingID = test_X.index
    test_X = test_X.reset_index(drop=True)
    test_X.drop(columns=["second_min"], inplace=True)

    # generate distance, velocity and angle features
    for col in test_X.columns:
        if col.startswith("second"):
            agg_method = col.split("_")[1]
            test_X["distance_" + agg_method] = test_X[col] * test_X["Speed_" + agg_method]
            test_X["velocity_x_" + agg_method] = test_X[col] * test_X["acceleration_x_" + agg_method]
            test_X["velocity_y_" + agg_method] = test_X[col] * test_X["acceleration_y_" + agg_method]
            test_X["velocity_z_" + agg_method] = test_X[col] * test_X["acceleration_z_" + agg_method]
            test_X["angle_x_" + agg_method] = test_X[col] * test_X["gyro_x_" + agg_method]
            test_X["angle_y_" + agg_method] = test_X[col] * test_X["gyro_y_" + agg_method]
            test_X["angle_z_" + agg_method] = test_X[col] * test_X["gyro_z_" + agg_method]
    
    sr = load("stackingRegressor.joblib")
    y_pred = sr.predict(test_X)

    output_df = pd.DataFrame({"bookingID": bookingID, "label": y_pred})
    output_df.to_csv("test_result.csv")