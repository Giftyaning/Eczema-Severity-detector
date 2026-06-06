# ============================================================

# ==== WHAT THIS CDE DOES
#This script reruns both models but converts all images to 
# grayscale first. I kept everything else identical so it 
# was a fair comparison. The purpose was to find out whether 
# the models rely on colour to make their decisions 
# and the results confirmed they do, because both models 
# performed worse without colour.

# FILE: grayscale_experiment.py
# Purpose: To investigate whether converting images to
#          grayscale affects model classification performance.
#          This tests whether models rely on colour cues
#          such as skin redness, or learn texture and
#          structural features that persist without colour.
#          Results are compared against the colour image
#          results from cnn_model.py and mobilenet_model.py
# ============================================================

import tensorflow as tf  # Imports TensorFlow — used to build, train and run both models
import numpy as np  # Imports NumPy — used for number calculations and handling prediction arrays
import matplotlib.pyplot as plt  # Imports Matplotlib — used to draw and save the accuracy and loss graphs
import seaborn as sns  # Imports Seaborn — used to draw the confusion matrix heatmaps
from sklearn.metrics import confusion_matrix, classification_report  # Imports tools to measure how well each model performed

# ─────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────

img_height = 224  # Sets image height to 224 pixels — must match what both models expect
img_width = 224   # Sets image width to 224 pixels
batch_size = 16   # Loads 16 images at a time — kept the same as the colour experiments for a fair comparison

# ─────────────────────────────────────────────────────────────
# GRAYSCALE PREPROCESSING FUNCTION
# Converts colour images to grayscale then back to 3 channels
# Why back to 3 channels? Because both CNN and MobileNetV2
# expect input shape (224, 224, 3) — they cannot accept
# single channel images without architectural changes.
# Converting grayscale back to 3 identical channels preserves
# the architecture whilst removing all colour information.
# ─────────────────────────────────────────────────────────────

def to_grayscale_3channel(image, label):  # Defines a reusable function that converts any image to grayscale while keeping 3 channels
    gray = tf.image.rgb_to_grayscale(image)  # Converts the colour image to grayscale — reduces from 3 colour channels down to 1 channel

    gray_3ch = tf.repeat(gray, 3, axis=-1)  # Copies the single grayscale channel 3 times to make it 3 channels again — this is needed because both models only accept 3-channel inputs. All 3 channels are identical so there is no colour information

    return gray_3ch, label  # Returns the grayscale 3-channel image and its label unchanged


# ─────────────────────────────────────────────────────────────
# PART 1: CNN WITH GRAYSCALE IMAGES
# ─────────────────────────────────────────────────────────────

print("=" * 60)  # Prints a divider line
print("PART 1: CNN MODEL — GRAYSCALE IMAGES")  # Prints a heading for Part 1
print("=" * 60)  # Prints another divider

train_data_cnn = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",  # Loads training images from the train folder
    image_size=(img_height, img_width),  # Resizes every image to 224x224
    batch_size=batch_size,  # Loads 16 images at a time
    shuffle=True  # Randomly shuffles training images each epoch so the model doesn't learn the order
)

val_data_cnn = tf.keras.utils.image_dataset_from_directory(
    "dataset/validation",  # Loads validation images from the validation folder
    image_size=(img_height, img_width),  # Resizes to 224x224
    batch_size=batch_size,  # Loads 16 at a time
    shuffle=False  # Does not shuffle — keeps a fixed order so predictions match the correct images
)

class_names = train_data_cnn.class_names  # Reads the folder names as class labels — high and low
print(f"Classes: {class_names}")  # Prints the class names to confirm they loaded correctly

train_data_cnn = train_data_cnn.map(lambda x, y: to_grayscale_3channel(x / 255.0, y))  # Normalises training images to 0-1 first, then converts to grayscale 3-channel
val_data_cnn = val_data_cnn.map(lambda x, y: to_grayscale_3channel(x / 255.0, y))  # Does the same normalisation and grayscale conversion for validation images

cnn_model = tf.keras.Sequential([  # Builds the CNN as a stack of layers — identical architecture to cnn_model.py for a fair comparison
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),  # First scanning layer — 32 filters detecting basic features like edges
    tf.keras.layers.MaxPooling2D(),  # Shrinks the image by keeping only the strongest features
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),  # Second scanning layer — 64 filters for more complex patterns
    tf.keras.layers.MaxPooling2D(),  # Shrinks again
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),  # Third scanning layer — 128 filters for even more complex features
    tf.keras.layers.MaxPooling2D(),  # Shrinks one final time
    tf.keras.layers.Flatten(),  # Converts 2D feature maps into a flat list of numbers
    tf.keras.layers.Dense(128, activation='relu'),  # Fully connected layer combining all features to make a decision
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer — produces a probability between 0 and 1
])

cnn_model.summary()  # Prints a summary of every layer and the total number of parameters

cnn_model.compile(
    optimizer='adam',  # Uses Adam optimiser — adjusts learning speed automatically
    loss='binary_crossentropy',  # Standard loss function for two-class problems
    metrics=['accuracy']  # Tracks accuracy during training
)

print("\nTraining CNN on grayscale images (10 epochs)...")  # Prints a message so you know training is starting
cnn_history = cnn_model.fit(
    train_data_cnn,  # Feeds the grayscale training images into the CNN
    validation_data=val_data_cnn,  # Tests on grayscale validation images after each epoch
    epochs=10  # Trains for 10 rounds
)

# ── CNN GRAYSCALE GRAPHS ─────────────────────────────────────

plt.figure(figsize=(12, 4))  # Creates a blank figure to hold both graphs side by side

plt.subplot(1, 2, 1)  # Places the next graph in position 1 of a 1-row 2-column layout
plt.plot(cnn_history.history['accuracy'], label='Train Accuracy', marker='o', color='blue')  # Draws the training accuracy line in blue
plt.plot(cnn_history.history['val_accuracy'], label='Val Accuracy', marker='o', color='orange')  # Draws the validation accuracy line in orange
plt.title('CNN Grayscale — Accuracy')  # Adds a title to the accuracy graph
plt.xlabel('Epoch')  # Labels the x-axis
plt.ylabel('Accuracy')  # Labels the y-axis
plt.legend()  # Adds a legend showing which line is training and which is validation
plt.grid(True)  # Adds a background grid for easier reading

plt.subplot(1, 2, 2)  # Places the next graph in position 2
plt.plot(cnn_history.history['loss'], label='Train Loss', marker='o', color='blue')  # Draws the training loss line
plt.plot(cnn_history.history['val_loss'], label='Val Loss', marker='o', color='orange')  # Draws the validation loss line
plt.title('CNN Grayscale — Loss')  # Adds a title to the loss graph
plt.xlabel('Epoch')  # Labels the x-axis
plt.ylabel('Loss')  # Labels the y-axis
plt.legend()  # Adds the legend
plt.grid(True)  # Adds the grid

plt.tight_layout()  # Adjusts spacing so the graphs don't overlap
plt.savefig('cnn_grayscale_accuracy_loss.png', dpi=150)  # Saves the graphs as a high resolution image file
plt.show()  # Displays the graphs on screen
print("✅ Graph saved: cnn_grayscale_accuracy_loss.png")  # Prints a confirmation message

# ── CNN GRAYSCALE CONFUSION MATRIX ───────────────────────────

y_pred_probs_cnn = cnn_model.predict(val_data_cnn)  # Runs all 40 grayscale validation images through the CNN and gets a probability for each
y_pred_cnn = (y_pred_probs_cnn > 0.5).astype(int).flatten()  # Converts probabilities to 0 or 1 — above 0.5 is high severity, below is low severity
y_true_cnn = np.concatenate([y for x, y in val_data_cnn], axis=0).astype(int).flatten()  # Collects the actual correct labels from the validation set

cm_cnn = confusion_matrix(y_true_cnn, y_pred_cnn)  # Compares true labels against predictions to build the confusion matrix

plt.figure(figsize=(6, 5))  # Creates a blank figure for the confusion matrix
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues',  # Draws the confusion matrix as a blue heatmap with numbers inside each box
            xticklabels=class_names, yticklabels=class_names)  # Labels the rows and columns with high and low
plt.title('CNN Grayscale — Confusion Matrix')  # Adds a title
plt.xlabel('Predicted Label')  # Labels the x-axis — what the model predicted
plt.ylabel('True Label')  # Labels the y-axis — what the correct answer was
plt.tight_layout()  # Adjusts spacing
plt.savefig('cnn_grayscale_confusion_matrix.png', dpi=150)  # Saves the confusion matrix as an image file
plt.show()  # Displays it on screen
print("✅ Confusion matrix saved: cnn_grayscale_confusion_matrix.png")  # Prints a confirmation

print("\n── CNN Grayscale Classification Report ──────────────")  # Prints a heading before the report
print(classification_report(y_true_cnn, y_pred_cnn, target_names=class_names))  # Prints precision, recall, F1-score and accuracy for each class


# ─────────────────────────────────────────────────────────────
# PART 2: MOBILENETV2 WITH GRAYSCALE IMAGES
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)  # Prints a blank line then a divider to start Part 2
print("PART 2: MOBILENETV2 MODEL — GRAYSCALE IMAGES")  # Prints a heading for Part 2
print("=" * 60)  # Prints another divider

train_data_mob = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",  # Loads training images for MobileNetV2
    image_size=(img_height, img_width),  # Resizes to 224x224
    batch_size=batch_size,  # Loads 16 at a time
    shuffle=True  # Shuffles training images
)

val_data_mob = tf.keras.utils.image_dataset_from_directory(
    "dataset/validation",  # Loads validation images for MobileNetV2
    image_size=(img_height, img_width),  # Resizes to 224x224
    batch_size=batch_size,  # Loads 16 at a time
    shuffle=False  # No shuffle — keeps fixed order for accurate confusion matrix
)

train_data_mob = train_data_mob.map(
    lambda x, y: to_grayscale_3channel(
        tf.keras.applications.mobilenet_v2.preprocess_input(x), y  # Applies MobileNetV2's specific preprocessing first to scale pixels to -1 to 1, then converts to grayscale 3-channel
    )
)
val_data_mob = val_data_mob.map(
    lambda x, y: to_grayscale_3channel(
        tf.keras.applications.mobilenet_v2.preprocess_input(x), y  # Same preprocessing applied to validation images
    )
)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),  # Tells MobileNetV2 the size and channels of images it will receive
    include_top=False,  # Removes MobileNetV2's original output layer — we add our own
    weights='imagenet'  # Loads pretrained weights from ImageNet — the knowledge from 1.2 million images
)
base_model.trainable = False  # Freezes all pretrained layers so they don't change — only our new layers will be trained

mob_model = tf.keras.Sequential([  # Builds the full MobileNetV2 model — identical to mobilenet_model.py for a fair comparison
    base_model,  # The frozen pretrained MobileNetV2 base
    tf.keras.layers.GlobalAveragePooling2D(),  # Converts the base output into a single flat vector by averaging
    tf.keras.layers.Dense(128, activation='relu'),  # Fully connected layer to learn eczema-specific patterns
    tf.keras.layers.Dropout(0.3),  # Randomly switches off 30% of neurons during training to prevent overfitting
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer — produces a probability between 0 and 1
])

mob_model.summary()  # Prints a summary of every layer and parameter count

mob_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Uses Adam with a small learning rate of 0.0001 so the pretrained weights aren't changed too aggressively
    loss='binary_crossentropy',  # Standard loss function for two-class problems
    metrics=['accuracy']  # Tracks accuracy during training
)

print("\nTraining MobileNetV2 on grayscale images (10 epochs)...")  # Prints a message so you know training is starting
mob_history = mob_model.fit(
    train_data_mob,  # Feeds grayscale training images into MobileNetV2
    validation_data=val_data_mob,  # Tests on grayscale validation images after each epoch
    epochs=10  # Trains for 10 rounds
)

# ── MOBILENETV2 GRAYSCALE GRAPHS ─────────────────────────────

plt.figure(figsize=(12, 4))  # Creates a blank figure for both graphs

plt.subplot(1, 2, 1)  # Position 1 of 2
plt.plot(mob_history.history['accuracy'], label='Train Accuracy', marker='o', color='blue')  # Training accuracy line
plt.plot(mob_history.history['val_accuracy'], label='Val Accuracy', marker='o', color='orange')  # Validation accuracy line
plt.title('MobileNetV2 Grayscale — Accuracy')  # Graph title
plt.xlabel('Epoch')  # X-axis label
plt.ylabel('Accuracy')  # Y-axis label
plt.legend()  # Adds the legend
plt.grid(True)  # Adds the grid

plt.subplot(1, 2, 2)  # Position 2 of 2
plt.plot(mob_history.history['loss'], label='Train Loss', marker='o', color='blue')  # Training loss line
plt.plot(mob_history.history['val_loss'], label='Val Loss', marker='o', color='orange')  # Validation loss line
plt.title('MobileNetV2 Grayscale — Loss')  # Graph title
plt.xlabel('Epoch')  # X-axis label
plt.ylabel('Loss')  # Y-axis label
plt.legend()  # Adds the legend
plt.grid(True)  # Adds the grid

plt.tight_layout()  # Prevents graphs overlapping
plt.savefig('mobilenet_grayscale_accuracy_loss.png', dpi=150)  # Saves the graphs as a high resolution image
plt.show()  # Displays on screen
print("✅ Graph saved: mobilenet_grayscale_accuracy_loss.png")  # Confirmation message

# ── MOBILENETV2 GRAYSCALE CONFUSION MATRIX ───────────────────

y_pred_probs_mob = mob_model.predict(val_data_mob)  # Runs all 40 grayscale validation images through MobileNetV2 and gets a probability for each
y_pred_mob = (y_pred_probs_mob > 0.5).astype(int).flatten()  # Converts probabilities to 0 or 1
y_true_mob = np.concatenate([y for x, y in val_data_mob], axis=0).astype(int).flatten()  # Collects the actual correct labels

cm_mob = confusion_matrix(y_true_mob, y_pred_mob)  # Builds the confusion matrix by comparing true labels against predictions

plt.figure(figsize=(6, 5))  # Creates a blank figure
sns.heatmap(cm_mob, annot=True, fmt='d', cmap='Greens',  # Draws the confusion matrix as a green heatmap — green used here to visually distinguish it from the CNN's blue one
            xticklabels=class_names, yticklabels=class_names)  # Labels rows and columns with high and low
plt.title('MobileNetV2 Grayscale — Confusion Matrix')  # Adds a title
plt.xlabel('Predicted Label')  # X-axis label
plt.ylabel('True Label')  # Y-axis label
plt.tight_layout()  # Adjusts spacing
plt.savefig('mobilenet_grayscale_confusion_matrix.png', dpi=150)  # Saves as a high resolution image
plt.show()  # Displays on screen
print("✅ Confusion matrix saved: mobilenet_grayscale_confusion_matrix.png")  # Confirmation message

print("\n── MobileNetV2 Grayscale Classification Report ──────")  # Heading before the report
print(classification_report(y_true_mob, y_pred_mob, target_names=class_names))  # Prints full precision, recall, F1-score breakdown for each class

# ─────────────────────────────────────────────────────────────
# PART 3: COMPARISON SUMMARY
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)  # Divider
print("GRAYSCALE EXPERIMENT — SUMMARY")  # Heading for the final summary
print("=" * 60)  # Divider
print(f"CNN Grayscale Final Val Accuracy:        {cnn_history.history['val_accuracy'][-1]*100:.1f}%")  # Prints the CNN's final validation accuracy from the last epoch as a percentage
print(f"MobileNetV2 Grayscale Final Val Accuracy: {mob_history.history['val_accuracy'][-1]*100:.1f}%")  # Prints MobileNetV2's final validation accuracy as a percentage
print("=" * 60)  # Divider
print("\nFiles saved:")  # Lists all the output files
print("  - cnn_grayscale_accuracy_loss.png")  # CNN accuracy and loss graph
print("  - cnn_grayscale_confusion_matrix.png")  # CNN confusion matrix
print("  - mobilenet_grayscale_accuracy_loss.png")  # MobileNetV2 accuracy and loss graph
print("  - mobilenet_grayscale_confusion_matrix.png")  # MobileNetV2 confusion matrix