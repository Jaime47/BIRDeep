import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import datetime

# Load class names
class_names = np.load("class_names.npy", allow_pickle=True).item()

# Load train data
train_df = pd.read_csv("train_images.csv")

# Assuming you have train_images folder with images
train_datagen = ImageDataGenerator(rescale=1./255)

# Assuming script_dir is the directory of your script
script_dir = os.path.dirname(os.path.abspath(__file__))
train_images = os.path.join(script_dir, 'train_images')

# Load and preprocess images
X = []
y = []

for index, row in train_df.iterrows():
    img_filename = row['image_path'][1:]  # Remove leading "/"
    img_path = os.path.join(train_images, img_filename)
    label = row['label'] - 1

    img = cv2.imread(img_path)
    
    # Ensure that the image has 3 channels
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0

    X.append(img)
    y.append(label)

# Convert lists to NumPy arrays
X_train = np.array(X)
y_train = np.array(y)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Model architecture
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.2))  # dropout 
model.add(Dense(128, activation='elu'))
model.add(Dropout(0.2))  # dropout 
model.add(Dense(128, activation='elu'))
model.add(Dropout(0.2))  # dropout 
model.add(Dense(200, activation='softmax'))  # Assuming 200 classes

# Compile the model
model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Load test data paths
test_df = pd.read_csv("test_images_path.csv")

# Assuming you have test_images folder with images
test_datagen = ImageDataGenerator(rescale=1./255)

test_images = os.path.join(script_dir, 'test_images')

# Load and preprocess test images
X_test = []

for index, row in test_df.iterrows():
    img_filename = row['image_path'][1:]  # Remove leading "/"
    img_path = os.path.join(test_images, img_filename)

    img = cv2.imread(img_path)
    
    # Ensure that the image has 3 channels
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img, (224, 224))
    img = img / 255.0

    X_test.append(img)

# Convert list to NumPy array
X_test = np.array(X_test)

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
