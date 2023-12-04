import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import models, layers
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomContrast, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1, l2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import datetime
from tensorflow.keras.applications import InceptionV3


## Pathing
path_to_code = '/Users/jaimeponsgarrido/Downloads/BIRDeep/code/'
path_to_sub = '/Users/jaimeponsgarrido/Downloads/BIRDeep/sub/'
train_folder = '/Users/jaimeponsgarrido/Downloads/BIRDeep/code/train_images/' 
test_folder = '/Users/jaimeponsgarrido/Downloads/BIRDeep/code/test_images/'
path_to_train_csv_file = '/Users/jaimeponsgarrido/Downloads/BIRDeep/code/train_images.csv' 
path_to_test_csv_file = '/Users/jaimeponsgarrido/Downloads/BIRDeep/code/test_images_path.csv'  
###
train = pd.read_csv(path_to_train_csv_file)
test = pd.read_csv(path_to_test_csv_file)
###
### Variables
batch_size = 32
img_height = 299
img_width = 299
###
train['label'] = train['label'] - 1

# Data loading
class_names = np.load(path_to_code + "class_names.npy", allow_pickle=True).item()

train_ds = tf.keras.utils.image_dataset_from_directory(
  train_folder,
  validation_split=0.2,
  subset="training",
  label_mode = 'int',
  labels = train['label'].tolist(),
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  train_folder,
  validation_split=0.2,
  subset="validation",
  label_mode = 'int',
  labels = train['label'].tolist(),
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
  test_folder,
  labels = None,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

AUTOTUNE = tf.data.AUTOTUNE
# Augmentation
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])
# Resizing
resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(299, 299),
  layers.Rescaling(1./255)
])
# Data augmentation / resizing routine
def prepare(ds, shuffle=False, augment=False):
    if shuffle:
        ds = ds.shuffle(1000)
    
    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                num_parallel_calls=AUTOTUNE)

    return ds.prefetch(buffer_size=AUTOTUNE)

train_ds = prepare(train_ds, shuffle=True, augment=True)
val_ds = prepare(val_ds)
test_ds = prepare(test_ds)

print('Message: End of data augmentation')

# Xception model
base_model = tf.keras.applications.xception.Xception(weights="imagenet",
                                                     include_top=False)

avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
dropout = Dropout(0.5)(avg) 
dense_layer = tf.keras.layers.Dense(200, activation="softmax", kernel_regularizer=l2(0.01))(dropout)
model = tf.keras.Model(inputs=base_model.input, outputs=dense_layer)

for layer in base_model.layers:
    layer.trainable = False

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy()
             ,optimizer = optimizer
             ,metrics = ['accuracy'])

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
             
model.summary()

### Model fit
history = model.fit(train_ds
         ,steps_per_epoch = len(train_ds)
         ,epochs = 12
         ,validation_data = val_ds
         ,validation_steps = int(0.25*len(val_ds))
         ,callbacks=[early_stopping])

# Make predictions on the test set
predictions = model.predict(test_ds)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)
print(predicted_labels)
# Update the 'label' column in the test dataframe with the predicted labels
results = pd.DataFrame({'id': range(1, 4001)})

results['label'] = predicted_labels

# Generate a timestamp for the filename
current_datetime = datetime.datetime.now()
timestamp = current_datetime.strftime("%Y%m%d_%H%M")
file_name = f"submission_{timestamp}.csv"

# Specify the full file path
file_path = os.path.join(path_to_sub, file_name)

# Save the final submission file
results.to_csv(file_path, index=False)
