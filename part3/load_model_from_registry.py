import os 
from argparse import ArgumentParser

import mlflow 
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# set mlflow enviroments
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] ="miniostorage"

# load model from mlflow
parser = ArgumentParser()
parser.add_argument("--model_name", dest = "model_name", type=str, default="sk_model")
parser.add_argument("--model_name_2", dest="model_name_2",type=str, default="sk_rf_model")
parser.add_argument("--run_id", dest="run_id", type=str)
args = parser.parse_args()

model_pipeline = mlflow.sklearn.load_model(f"runs:/{args.run_id}/{args.model_name}")
rf_model_pipeline = mlflow.sklearn.load_model(f"runs:/{args.run_id}/{args.model_name_2}")

df = pd.read_csv("data.csv")
X = df.drop(["id","timestamp","target"], axis = "columns")
y = df["target"]
X_train, X_valid, y_train, y_valid = train_test_split(X,y, train_size=0.8, random_state=1998)

train_pred = model_pipeline.predict(X_train)
valid_pred = model_pipeline.predict(X_valid)
rf_train_pred = rf_model_pipeline.predict(X_train)
rf_valid_pred = rf_model_pipeline.predict(X_valid)

train_acc = accuracy_score(y_true=y_train, y_pred=train_pred)
valid_acc = accuracy_score(y_true=y_valid, y_pred=valid_pred)
rf_train_acc = accuracy_score(y_true=y_train, y_pred=rf_train_pred)
rf_valid_acc = accuracy_score(y_true=y_valid, y_pred=rf_valid_pred)

print("Train Accuracy :", train_acc)
print("Valid Accuracy :", valid_acc)
print("RF_Train Accuracy :", rf_train_acc)
print("RF_Valid Accuracy :", rf_valid_acc)