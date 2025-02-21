import logging
import pandas as pd
import numpy as np
from zenml import step
from src.evaluation import MSE,RMSE,R2
from typing_extensions import Annotated
from typing import Tuple
from sklearn.base import RegressorMixin


@step
def evaluate_model(model:RegressorMixin,x_test:pd.DataFrame,y_test:pd.DataFrame)-> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:

    try:
    
        prediction = model.predict(x_test)

        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test,prediction)

        r2_class = R2()

        r2 = r2_class.calculate_scores(y_test,prediction)

        rmse_class = RMSE()

        rmse = rmse_class.calculate_scores(y_test,prediction)

        return r2,rmse
    
    except Exception as e:
        logging.error("Error while evaluating the model")
        raise e

