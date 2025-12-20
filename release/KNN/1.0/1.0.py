import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('updated_cleaned_patient_data.csv')

feature_cols = [
    'FBS', 'BMI', 'Diabetes', 'age', 'hypertension',
    'vegetarian (1= yes, 0=no)', 'Exercise (min/week)',
    'Cholesterol', 'Living_Area_Code', 'Marriage_Status_Code'
]

X = data[feature_cols].fillna(0)
y = data['stroke']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# model = KNeighborsClassifier(n_neighbors=3, weights='uniform', p=1)
# p=1 หมายถึง Manhattan Distance (วัดระยะแบบเดินตามบล็อกตาราง หรือ ผลรวมค่าสัมบูรณ์)

# model = KNeighborsClassifier(n_neighbors=3, weights='uniform', p=2)
# p=2 หมายถึง Euclidean Distance (วัดระยะแบบลากเส้นตรงจุดต่อจุด หรือ สูตรพีทาโกรัส)

# model = KNeighborsClassifier(n_neighbors=3, metric='hamming')
# metric='hamming' หมายถึง Hamming Distance (เหมาะกับข้อมูลหมวดหมู่/Binary: วัดระยะโดยการนับจำนวนตำแหน่งที่ค่า "ไม่เหมือนกัน")

model = KNeighborsClassifier(n_neighbors=3,weights='uniform',p=1)
model.fit(X_train_scaled, y_train)
train_pred = model.predict(X_train_scaled)
test_pred = model.predict(X_test_scaled)

print("Train Accuracy:", accuracy_score(y_train, train_pred))
print("Test Accuracy:", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred))