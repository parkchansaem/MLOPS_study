import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# get_data
X,y = load_iris(return_X_y=True, as_frame=True)
X_train, X_valid, y_train, y_valid = train_test_split(X,y, train_size=0.8, random_state=1998)

# model development and train
scaler = StandardScaler()
classifier = SVC()
RF_classifier = RandomForestClassifier()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_valid = scaler.transform(X_valid)
classifier.fit(scaled_X_train,y_train)
RF_classifier.fit(scaled_X_train,y_train)

train_pred = classifier.predict(scaled_X_train)
valid_pred = classifier.predict(scaled_X_valid)
RF_train_pred = RF_classifier.predict(scaled_X_train)
RF_valid_pred = RF_classifier.predict(scaled_X_valid)

train_acc = accuracy_score(y_true=y_train, y_pred = train_pred)
valid_acc = accuracy_score(y_true=y_valid, y_pred = valid_pred)
RF_train_acc = accuracy_score(y_true = y_train, y_pred = RF_train_pred)
RF_valid_acc = accuracy_score(y_true = y_valid, y_pred = RF_valid_pred)

print("Train Accuracy : ", train_acc)
print("Valid Accuracy : ", valid_acc)
print("RF Train Accuracy :", RF_train_acc)
print("RF_Valid Accuracy :", RF_valid_acc)

# save model 
joblib.dump(scaler,"scaler.joblib")
joblib.dump(classifier, "classifier.joblib")
joblib.dump(RF_classifier, "RF_classifier.joblib")




