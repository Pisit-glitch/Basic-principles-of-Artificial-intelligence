import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

# 1. โหลดข้อมูล
data_path = 'updated_cleaned_patient_data.csv'
data = pd.read_csv(data_path)

# 2. เลือกฟีเจอร์ (ใช้เป็นตัวเลขตรงๆ ไม่ต้องแปลงเป็น Text)
feature_cols = [
    'FBS', 'BMI', 'Diabetes', 'age', 'hypertension',
    'vegetarian (1= yes, 0=no)', 'Exercise (min/week)',
    'Cholesterol', 'Living_Area_Code', 'Marriage_Status_Code'
]

X = data[feature_cols]
y = data['stroke']

# แบ่งข้อมูล (ใช้ stratify=y เพื่อรักษาสัดส่วนคนป่วยใน Train/Test ให้เท่ากัน)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. สร้าง Pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')), # เติมค่าที่หายไป (ถ้ามี) ด้วยค่าเฉลี่ย
    ('scaler', StandardScaler()),                # ปรับสเกลข้อมูลให้มาตรฐาน (สำคัญมากสำหรับ GaussianNB)
    ('clf', GaussianNB())                        # ตัวพระเอกของเรา
])

# 4. กำหนดค่าที่ต้องการจูน (Hyperparameter Tuning)
# var_smoothing: ช่วยลดปัญหากรณีข้อมูลมีความแปรปรวนแปลกๆ
param_grid = {
    'clf__var_smoothing': np.logspace(0, -9, num=100) # ไล่เช็ก 100 ค่า ตั้งแต่ 1.0 ถึง 0.000000001
}

# 5. เริ่มการค้นหา (Grid Search)
print("Starting Grid Search with GaussianNB...")
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5, 
    scoring='f1',  # ใช้ f1-score เป็นเกณฑ์ตัดสิน (เพื่อความสมดุลระหว่างความแม่นและการจับคนป่วย)
    n_jobs=-1, 
    verbose=1
)

grid_search.fit(X_train, y_train)

# 6. แสดงผลลัพธ์
print("\n" + "="*40)
print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")
print("Best Parameters:")
print(grid_search.best_params_)
print("="*40 + "\n")

# ทดสอบกับ Test Set
best_model = grid_search.best_estimator_
test_pred = best_model.predict(X_test)

print("Final Test Accuracy:", accuracy_score(y_test, test_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, test_pred))