import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from  sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


train_csv = "Train.csv"  # Path to train CSV
data = pd.read_csv(train_csv)
print(data.head())

images = []
labels = []

for i, row in data.iterrows():
    img_path = os.path.join("Train", str(row['ClassId']), row['Path'].split('/')[-1])
    img = cv2.imread(img_path)
    img = cv2.resize(img, (30,30))
    images.append(img)
    labels.append(row['ClassId'])

images = np.array(images, dtype='float32') / 255.0
labels = np.array(labels)

num_classes = len(np.unique(labels))
labels = to_categorical(labels, num_classes)

X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

print("Training Images shape:", X_train.shape)
print("Validation Images shape:", X_val.shape)

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1
)
datagen.fit(X_train)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(30,30,3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

from sklearn.utils.class_weight import compute_class_weight

# Convert one-hot labels back to integers
y_integers = np.argmax(y_train, axis=1)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_integers),
    y=y_integers
)

class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=15,
    validation_data=(X_val, y_val),
    class_weight=class_weights
)

val_loss, val_acc = model.evaluate(X_val, y_val)
print("Validation Accuracy:", val_acc)
y_pred=model.predict(X_val)
y_pred_classes=np.argmax(y_pred,axis=1)
y_true=np.argmax(y_val,axis=1)
cm=confusion_matrix(y_true,y_pred_classes)
plt.figure(figsize=(15,15))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.xlabel("predicted")
plt.ylabel("True")
plt.show()

model.save("traffic_sign_model_augmented.h5")
print("Model saved as traffic_sign_model_augmented.h5")

test_data = pd.read_csv('Test.csv')

test_images = []
test_labels = []

for i, row in test_data.iterrows():
    img_path =row['Path']
    img = cv2.imread(img_path)
    img = cv2.resize(img, (30,30))
    test_images.append(img)
    test_labels.append(row['ClassId'])

test_images = np.array(test_images, dtype='float32') / 255.0
test_labels = np.array(test_labels)
test_labels_cat = to_categorical(test_labels, num_classes)

test_loss, test_acc = model.evaluate(test_images, test_labels_cat)
print("Real Test Set Accuracy:", test_acc)

meta=pd.read_csv('Meta.csv')
y_pred_real=model.predict(test_images)
y_pred_classes_real=np.argmax(y_pred_real,axis=1)
for i in range(5):
    pred_index = y_pred_classes_real[i]
    true_index =test_labels[i]
    print(f"Image {i}: Predicted={pred_index}, True={true_index}")
