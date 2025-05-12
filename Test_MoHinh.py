import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input

# 1. Load model đã huấn luyện
model_path = 'models/model.h5'  # Đường dẫn tới mô hình đã lưu
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Mô hình không tồn tại tại đường dẫn: {model_path}")
model = load_model(model_path)

# 2. Khai báo các lớp
classes = ['la_khoe', 'suong_mai', 'than_thu', 'vi_rus']

# 3. Hàm xử lý hình ảnh đầu vào
def preprocess_image(image_path):
    """Đọc và xử lý hình ảnh đầu vào"""
    img = cv2.imread(image_path)  # Đọc ảnh
    if img is None:
        raise FileNotFoundError(f"Hình ảnh không tồn tại tại đường dẫn: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB
    img = cv2.resize(img, (224, 224))  # Resize về kích thước phù hợp với mô hình
    img = preprocess_input(img)  # Tiền xử lý với preprocess_input của MobileNet
    img = np.expand_dims(img, axis=0)  # Thêm chiều batch
    return img

# 4. Hàm dự đoán
def predict(image_path):
    """Dự đoán lớp của một hình ảnh"""
    img = preprocess_image(image_path)  # Tiền xử lý ảnh
    predictions = model.predict(img)  # Dự đoán
    predicted_class = np.argmax(predictions)  # Lấy chỉ số lớp có xác suất cao nhất
    confidence = predictions[0][predicted_class]  # Lấy xác suất của lớp dự đoán
    return classes[predicted_class], confidence

# 5. Kiểm tra trên hình ảnh mẫu
image_path = 'test/anh6.jpg'  # Đường dẫn tới hình ảnh để kiểm tra
if os.path.exists(image_path):
    predicted_class, confidence = predict(image_path)
    print(f"Hình ảnh: {image_path}")
    print(f"Dự đoán: {predicted_class} (Độ tin cậy: {confidence:.2f})")
else:
    print(f"Hình ảnh không tồn tại tại đường dẫn: {image_path}")
