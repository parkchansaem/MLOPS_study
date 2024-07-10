import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# reproduce data
X,y = load_iris(return_X_y=True, as_frame=True)
X_train,X_valid,y_train,y_valid = train_test_split(X,y,train_size=0.8, random_state=1998)

#load model
scaler_load = joblib.load("scaler.joblib")
classifier_load = joblib.load("classifier.joblib")
RF_classifier_load = joblib.load("RF_classifier.joblib")

# valiate
scaled_X_train = scaler_load.transform(X_train)
scaled_X_valid = scaler_load.transform(X_valid)

load_train_pred = classifier_load.predict(scaled_X_train)
load_valid_pred = classifier_load.predict(scaled_X_valid)
RF_load_train_pred = RF_classifier_load.predict(scaled_X_train)
Rf_load_valid_pred = RF_classifier_load.predict(scaled_X_valid)

load_train_acc = accuracy_score(y_true = y_train, y_pred = load_train_pred)
load_valid_acc = accuracy_score(y_true = y_valid, y_pred = load_valid_pred)
RF_load_train_acc = accuracy_score(y_true = y_train, y_pred = RF_load_train_pred)
Rf_load_valid_acc = accuracy_score(y_true = y_valid, y_pred = Rf_load_valid_pred)

print("Load Model Train Accuracy : ", load_train_acc)
print("Load Model Valid Accuracy : ", load_valid_acc)
print("Load RF Model Train Accuracy :", RF_load_train_acc)
print("Load RF Model Valid Accuracy :", Rf_load_valid_acc)