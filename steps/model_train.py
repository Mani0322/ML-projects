import logging
import pandas as pd
import numpy as np
from zenml import step
from src.model_dev import LinearRegressionModel
from .config import ModelNameConfig


@step


def train_model(X_train,X_test,y_train,y_test,config:ModelNameConfig):

    try:

        model = None
        if config.model_name == "LinearRegression":
            model = LinearRegressionModel()
            trained_model = model.train(X_train,y_train)
            return trained_model
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as e:
        logging.error("Error in training model {}".format(e))
        raise e
