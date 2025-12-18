import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 1. ตั้งค่าขนาดของภาพ (ปรับตัวเลขได้ถ้าภาพเล็กไป)
plt.figure(figsize=(25, 12))

# 2. วาดกราฟ
plot_tree(
    model,
    feature_names=X.columns,      # ใส่ชื่อคอลัมน์เพื่อให้รูว่าโหนดนั้นเช็คค่าอะไร
    class_names=['No Stroke', 'Stroke'], # ใส่ชื่อผลลัพธ์ (0=No Stroke, 1=Stroke)
    filled=True,                  # ระบายสีตามความมั่นใจ (สีเข้ม=มั่นใจมาก)
    rounded=True,                 # ทำให้กล่องมน ดูง่ายขึ้น
    fontsize=10                   # ขนาดตัวอักษร
)

# 3. แสดงผล
plt.title("Decision Tree Visualization for Stroke Prediction")
plt.show()