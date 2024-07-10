import os
from argparse import ArgumentParser

import mlflow
import pandas as pd
import pymysql
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# set mlflow enmvironments
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"

db_config = { 
    'user' :"chanseam" ,
    'password' : 'qwe123!@#',
    'host' : "192.168.0.25",
    'port' : 4000,
    'database': 'test'
    }
db_connect = pymysql.connect(**db_config)
df = pd.read_sql("SELECT * FROM iris_data ORDER BY id DESC LIMIT 100 ; ",db_connect)
X = df.drop(["id","timestamp", "target"],axis="columns")
y = df['target']
X_train,X_valid, y_train, y_valid = train_test_split(X,y , train_size=0.8, random_state=1998)

model_pipeline = Pipeline([("scaler",StandardScaler()),("svc",SVC())])
rf_model_pipeline = Pipeline([("scaler",StandardScaler()),("rf",RandomForestClassifier())])
model_pipeline.fit(X_train,y_train)
rf_model_pipeline.fit(X_train,y_train)

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

# 3. save model
parser = ArgumentParser()
parser.add_argument("--model-name", dest="model_name", type=str, default="sk_model")
parser.add_argument("--model-name_2", dest="model_name_2", type=str, default="sk_rf_model")
args = parser.parse_args()

mlflow.set_experiment("new-exp")

signature = mlflow.models.signature.infer_signature(model_input=X_train, model_output=train_pred)
input_sample = X_train.iloc[:10]


with mlflow.start_run() as run:
    run_id = run.info.run_id
    mlflow.log_metrics({"train_acc": train_acc, "valid_acc": valid_acc})
    mlflow.log_metrics({"rf_train_acc": rf_train_acc, "rf_valid_acc": rf_valid_acc})
    mlflow.sklearn.log_model(
        sk_model = model_pipeline,
        artifact_path = args.model_name,
        signature = signature,
        input_example=input_sample,
    )
    mlflow.sklearn.log_model(
        sk_model = rf_model_pipeline,
        artifact_path = args.model_name_2,
        signature = signature,
        input_example=input_sample,
    )

# 4. save data
df.to_csv("data.csv", index=False)