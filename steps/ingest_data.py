import logging
import pandas as pd
import numpy as np
from zenml import step


class IngestData:
    def __init__(self,datapath):
        self.datapath = datapath

    def get_data(self):
        logging.info(f"ingesting data from {self.datapath}")
        return pd.read_csv(self.datapath)
    

@step
def ingest_df(datapath):
    try:
        data = IngestData(datapath)
        df = data.get_data()
        return df
    except Exception as e:
        logging.info(f"Error while ingesting data {e}")
        raise e