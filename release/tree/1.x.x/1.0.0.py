import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

data_path = 'updated_cleaned_patient_data.csv'
data = pd.read_csv(data_path)

cols_to_drop = [ 'Living_Area_Code', 'Marriage_Status_Code']
data.drop(columns=cols_to_drop, inplace=True, errors='ignore')

data_encoded = pd.get_dummies(data, drop_first=True)

X = data_encoded.drop('stroke', axis=1)
y = data_encoded['stroke']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=50, 
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("Train Accuracy :", accuracy_score(y_train, train_pred))
print("Validation Accuracy  :", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred))