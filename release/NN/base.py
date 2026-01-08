import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models

# ==========================================
# 0. ปรับตั้งค่า Parameter ตรงนี้ได้เลย
# ==========================================
LEARNING_RATE = 0.001       # Learning Rate (η)
EPOCHS = 50                # Epochs
BATCH_SIZE = 32            # Batch Size
ACTIVATION = 'relu'        # Activation Function (เช่น 'relu', 'tanh', 'sigmoid')
DROPOUT_RATE = 0.2         # Dropout Rate
# เราสามารถกำหนดจำนวน Node ในแต่ละชั้นได้ที่นี่
HIDDEN_UNITS_L1 = 32       
HIDDEN_UNITS_L2 = 16       
# เลือก Optimizer: tf.keras.optimizers.Adam, SGD, RMSprop
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# = :=========================================

# 1. โหลดข้อมูล
df = pd.read_csv('updated_cleaned_patient_data.csv')
df = df.fillna(df.mean())

# 2. แยก Features (X) และ Target (y)
X = df.drop('stroke', axis=1)
y = df['stroke']

# 3. แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. สร้างโครงสร้าง Neural Network
model = models.Sequential([
    # Input Layer และ Hidden Layer ที่ 1 (Hidden Units/Activation)
    layers.Dense(HIDDEN_UNITS_L1, activation=ACTIVATION, input_shape=(X_train.shape[1],)),
    layers.Dropout(DROPOUT_RATE), # Dropout Rate
    
    # Hidden Layer ที่ 2 (Number of Hidden Layers สามารถเพิ่ม/ลด layers.Dense ได้ตรงนี้)
    layers.Dense(HIDDEN_UNITS_L2, activation=ACTIVATION),
    
    # Output Layer
    layers.Dense(1, activation='sigmoid')
])

# 6. Compile โมเดล (Optimizer)
model.compile(optimizer=OPTIMIZER,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 7. เทรนโมเดล (Epochs / Batch Size)
history = model.fit(X_train, y_train, 
                    epochs=EPOCHS, 
                    batch_size=BATCH_SIZE, 
                    validation_split=0.2,
                    verbose=1)

# 8. วัดผลโมเดล
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nAccuracy on Test Set: {accuracy*100:.2f}%")