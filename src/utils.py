import os
import sys
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Saves a python object to a specific path using dill/pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Automatically trains multiple models, performs Hyperparameter tuning, 
    and returns a report of R2 scores.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            # Automatic Hyperparameter Tuning (Grid Search)
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Set the model to use the best parameters found
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Make Predictions
            y_test_pred = model.predict(X_test)

            # Get R2 Score for the test data
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def model_metrics(true, predicted):
    try:
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mse)
        r2_square = r2_score(true, predicted)
        return mae, rmse, r2_square
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)