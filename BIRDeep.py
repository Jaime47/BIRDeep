import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomContrast, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1, l2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import datetime

# Load class names
class_names = np.load("class_names.npy", allow_pickle=True).item()

# Load and preprocess images
def load_and_preprocess_images(data_df, image_folder):
    X = []
    y = []

    for index, row in data_df.iterrows():
        img_filename = row['image_path'][1:]  # Remove leading "/"
        img_path = os.path.join(image_folder, img_filename)
        label = row['label'] - 1

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (299, 299))
        img = img / 255.0

        X.append(img)
        y.append(label)

    return np.array(X), np.array(y)

# Load train data
train_df = pd.read_csv("train_images.csv")
script_dir = os.path.dirname(os.path.abspath(__file__))
train_images = os.path.join(script_dir, 'train_images')
X_train, y_train = load_and_preprocess_images(train_df, train_images)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Data Augmentation - this is not working yet
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode="horizontal", seed=42),
    tf.keras.layers.RandomRotation(factor=0.05, seed=42),
    tf.keras.layers.RandomContrast(factor=0.2, seed=42)
])

# Model architecture
base_model = tf.keras.applications.xception.Xception(weights="imagenet",
                                                     include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
dropout = Dropout(0.5)(avg) 
dense_layer = tf.keras.layers.Dense(200, activation="softmax", kernel_regularizer=l2(0.01))(dropout)
model = tf.keras.Model(inputs=base_model.input, outputs=dense_layer)

for layer in base_model.layers:
    layer.trainable = False

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with early stopping
model.fit(
    X_train,
    y_train,
    epochs=20,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping] 
)

# Load test data paths
test_df = pd.read_csv("test_images_path.csv")
test_images = os.path.join(script_dir, 'test_images')
X_test, _ = load_and_preprocess_images(test_df, test_images)

# Make predictions on the test set
predictions = model.predict(X_test)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Update the 'label' column in the test dataframe with the predicted labels
test_df['label'] = predicted_labels

selected_columns = ['id', 'label']
test_df_selected = test_df[selected_columns]

# Generate a timestamp for the filename
current_datetime = datetime.datetime.now()
timestamp = current_datetime.strftime("%Y%m%d_%H%M")
file_name = f"submission_{timestamp}.csv"

# Define the folder for submissions
submissions_folder = os.path.join(script_dir, 'submissions')

# Specify the full file path
file_path = os.path.join(submissions_folder, file_name)

# Save the final submission file
test_df_selected.to_csv(file_path, index=False)
