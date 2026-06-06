## To load and preview the dataset.
import tensorflow as tf  # Imports TensorFlow — the main library used to load and handle the image dataset

## Image dimension settings
img_height = 224  # Sets the height of every image to 224 pixels — both models require this exact size
img_width = 224   # Sets the width of every image to 224 pixels
batch_size = 16   # Sets how many images are loaded and processed at once — 16 at a time saves memory

## To load training data
train_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",                        # Tells TensorFlow where the training images are stored on the computer
    image_size=(img_height, img_width),     # Automatically resizes every training image to 224x224 when loading
    batch_size=batch_size                   # Loads 16 training images at a time instead of all at once
)

## To load validation data
val_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/validation",                   # Tells TensorFlow where the validation images are stored
    image_size=(img_height, img_width),     # Resizes every validation image to 224x224 — same as training
    batch_size=batch_size                   # Loads 16 validation images at a time
)

## To print class names
print("Classes:", train_data.class_names)  # Reads the subfolder names and prints them as the class labels — should print high and low