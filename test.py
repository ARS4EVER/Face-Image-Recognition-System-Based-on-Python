import os
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 调整分辨率的函数
def resize_images(images, new_size):
    if len(images.shape) == 1:  # 单个图像
        img_resized = Image.fromarray((images * 256).astype('uint8').reshape(57, 47))
        img_resized = img_resized.resize(new_size, Image.LANCZOS)
        return np.array(img_resized).flatten() / 256
    else:  # 批量图像
        resized_images = []
        for img in images:
            img_resized = Image.fromarray((img * 256).astype('uint8').reshape(57, 47))
            img_resized = img_resized.resize(new_size, Image.LANCZOS)
            resized_images.append(np.asarray(img_resized).flatten() / 256)
        return np.array(resized_images)

# 定义CNN模型
def create_model(input_shape):
    model = tf.keras.Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D(pool_size=(2, 2), padding='same'),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), padding='same'),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(40, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 读取数据
with open('D:/桌面/Face-recognition--master/Face-recognition--master/faces_data.pkl', 'rb') as read_file:
    faces = pickle.load(read_file)
    labels = pickle.load(read_file)

# 选择一个分辨率
resolution = (32, 30)  
input_shape = (resolution[1], resolution[0], 1)

# 准备训练数据
X_train = faces[:-1]  # 除了最后一个样本外的所有样本用于训练
y_train = labels[:-1]
X_test = faces[-1:]   # 最后一个样本用于测试
y_test = labels[-1:]

# 调整分辨率
X_train_resized = resize_images(X_train, resolution)
X_test_resized = resize_images(X_test[0], resolution)  # 注意这里是单个样本

# 创建模型
model = create_model(input_shape)

# 训练模型并保存历史记录
history = model.fit(
    X_train_resized.reshape(-1, resolution[1], resolution[0], 1),
    y_train,
    epochs=10,
    batch_size=20,
    verbose=1,
    validation_split=0.2  # 添加验证集
)

# 进行预测
X_test_resized = X_test_resized.reshape(1, resolution[1], resolution[0], 1)
prediction = model.predict(X_test_resized)
predicted_class = np.argmax(prediction)
actual_class = np.argmax(y_test)

print(f"预测的类别: {predicted_class}")
print(f"实际的类别: {actual_class}")


plt.figure(figsize=(5, 5))
plt.imshow(X_test[0].reshape(57, 47), cmap='gray')
plt.title(f'Test Image\nPredicted: {predicted_class}, Actual: {actual_class}')
plt.axis('off')
plt.show()

# 添加模型评估可视化
plt.figure(figsize=(12, 4))

# 绘制训练损失和验证损失
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制训练准确率和验证准确率
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
