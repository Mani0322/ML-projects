import logging
import pandas as pd
import numpy as np
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin


name ="LinearRegression"

@step



def train_model(X_train:pd.DataFrame,X_test:pd.DataFrame,y_train:pd.Series,y_test:pd.Series,name)-> RegressorMixin:

    try:

        model = None
        if name == "LinearRegression":
            model = LinearRegressionModel()
            trained_model = model.train(X_train,y_train)
            return trained_model
        else:
            raise ValueError("Model {} not supported".format(name))
    except Exception as e:
        logging.error("Error in training model {}".format(e))
        raise e
