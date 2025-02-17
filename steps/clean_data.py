import pandas as pd
import numpy as np
from zenml import step
import logging
from src.data_cleaning import DataCleaning,DataDivideStrategy,DataPreprocessStrategy



@step
def clean_df(df):
    try:
        process_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df,process_strategy)
        processed_data = data_cleaning.handle_data()
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data,divide_strategy)
        X_train,X_test,y_train,y_test = data_cleaning.handle_data()
        logging.info("data cleaning completed")
        return X_train,X_test,y_train,y_test
        
    except Exception as e:
        logging.error(f"Error while cleaning data:{e}")
        raise e