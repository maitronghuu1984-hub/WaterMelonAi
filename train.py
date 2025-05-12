import os
import cv2
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers

# 1. Định nghĩa tham số
n_class = 4
n_epochs = 10
batch_size = 64
data_folder = "data"

# Tạo thư mục lưu model nếu chưa tồn tại
os.makedirs('models', exist_ok=True)

# 2. Build model
def get_model():
    # Tạo base model
    base_model = MobileNet(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    # Tạo model chính
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(512, activation='relu')(x)
    outs = Dense(n_class, activation='softmax')(x)

    # Đóng băng các layer của base_model
    for layer in base_model.layers:
        layer.trainable = False

    model = Model(inputs=base_model.inputs, outputs=outs)
    return model

model = get_model()
model.summary()

# 3. Make data
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Đúng cú pháp
    rotation_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.3,
    zoom_range=0.5,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2
)
train_generator = train_datagen.flow_from_directory(
    data_folder,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_folder,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

classes = train_generator.class_indices
print(f"Classes: {classes}")
classes = list(classes.keys())

# 4. Train model
model.compile(optimizer=optimizers.SGD(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

# Lưu model tốt nhất dưới dạng .keras hoặc .h5
checkpoint = ModelCheckpoint('models/best.h5', monitor='val_loss', save_best_only=True, mode='auto')

callback_list = [checkpoint]

step_train = train_generator.n // batch_size
step_val = validation_generator.n // batch_size

# Sử dụng fit (thay vì fit_generator, đã bị thay thế)
model.fit(
    train_generator,
    steps_per_epoch=step_train,
    validation_data=validation_generator,
    validation_steps=step_val,
    callbacks=callback_list,
    epochs=n_epochs
)

# 5. Lưu model hoàn chỉnh
model.save('models/model.h5')
