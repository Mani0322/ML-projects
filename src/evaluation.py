import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score,root_mean_squared_error


class Evaluation(ABC):

    @abstractmethod
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        pass



class MSE(Evaluation):


    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        try:
            logging.info("calculating MSE")
            mse = mean_squared_error(y_true,y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error(f"Error while evaluating model")
            raise e
        

class R2(Evaluation):

    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        try:
            logging.info("calculating R2 Score")
            r2 = r2_score(y_true,y_pred)
            logging.info("R2 Score: {}".format(r2))
            return r2
        except Exception as e:
            logging.error(f"Error while evaluating model")
            raise e
        
class RMSE(Evaluation):

    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        try:
            logging.info("calculating RMSE")
            rmse = root_mean_squared_error(y_true,y_pred)
            logging.info("RMSE: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error(f"Error while evaluating model")
            raise e

     
        



        