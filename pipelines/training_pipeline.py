from zenml import pipeline

from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model
name ="LinearRegression"

@pipeline(enable_cache=False)
def train_pipeline(datapath):
    df = ingest_df(datapath)
    X_train,X_test,y_train,y_test = clean_df(df)
    model = train_model(X_train,X_test,y_train,y_test,name)
    r2,rmse = evaluate_model(model,X_test,y_test)

