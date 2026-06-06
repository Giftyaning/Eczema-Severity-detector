# ============================================================

# === WHAT THIS SCRIPT DOES 
# This script counts the images in each class folder to check for 
# imbalance, and it runs both models on the validation set to identify 
# exactly which images were misclassified and displays them visually."

# FILE: dataset_analysis.py
# Purpose 1: Count images in each class folder to identify
#            which class has fewer images (class imbalance)
# Purpose 2: Show which images were correctly and incorrectly
#            classified by both models
# ============================================================

import os  # Imports os — used to navigate folders and check if they exist on the computer
import numpy as np  # Imports NumPy — used for number calculations and handling arrays of predictions
import matplotlib.pyplot as plt  # Imports Matplotlib — used to display and save the misclassified image grids
import tensorflow as tf  # Imports TensorFlow — used to load images and build/train both models

# ─────────────────────────────────────────────────────────────
# PART 1: COUNT IMAGES IN EACH CLASS FOLDER
# ─────────────────────────────────────────────────────────────

print("=" * 55)  # Prints a line of 55 equal signs as a visual divider in the terminal
print("DATASET IMAGE COUNT ANALYSIS")  # Prints a heading so it's clear what this section does
print("=" * 55)  # Prints another divider line

dataset_path = "dataset"  # Sets the path to the main dataset folder
splits = ["train", "validation"]  # The two subfolders to check — training and validation
classes = ["high", "low"]  # The two class folders inside each split — high severity and low severity

for split in splits:  # Loops through train first, then validation
    print(f"\n📂 {split.upper()} SET:")  # Prints which set is being checked e.g. TRAIN SET
    split_total = 0  # Starts a counter at zero to count total images in this split
    counts = {}  # Creates an empty dictionary to store the count for each class

    for cls in classes:  # Loops through high severity first, then low severity
        folder_path = os.path.join(dataset_path, split, cls)  # Builds the full folder path e.g. dataset/train/high

        if os.path.exists(folder_path):  # Checks if that folder actually exists before trying to open it
            image_files = [
                f for f in os.listdir(folder_path)  # Lists every file in the folder
                if f.lower().endswith((".jpg", ".jpeg", ".png"))  # Only counts image files, ignoring anything else
                and not f.startswith(".")  # Ignores hidden system files that start with a dot
            ]
            count = len(image_files)  # Counts how many valid image files were found
            counts[cls] = count  # Stores the count for this class in the dictionary
            split_total += count  # Adds this class count to the running total
            print(f"   {cls.upper()} severity: {count} images")  # Prints the count e.g. HIGH severity: 79 images
        else:
            print(f"   {cls.upper()}: FOLDER NOT FOUND")  # Warns if the folder is missing instead of crashing

    print(f"   TOTAL: {split_total} images")  # Prints the total image count for this split

    if len(counts) == 2:  # Only runs the balance check if both classes were found
        diff = abs(counts["high"] - counts["low"])  # Calculates the difference between high and low image counts
        if diff == 0:
            print(f"   ✅ Classes are BALANCED")  # Prints a success message if both classes have equal images
        else:
            minority = "high" if counts["high"] < counts["low"] else "low"  # Identifies which class has fewer images
            majority = "low" if minority == "high" else "high"  # Identifies which class has more images
            print(f"   ⚠️  IMBALANCED by {diff} image(s)")  # Warns that the classes are not equal
            print(f"   ⚠️  {minority.upper()} has fewer images ({counts[minority]}) than {majority.upper()} ({counts[majority]})")  # Shows exactly which class is smaller and by how much

print("=" * 55)  # Prints a final divider to close Part 1

# ─────────────────────────────────────────────────────────────
# PART 2: MISCLASSIFICATION ANALYSIS
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 55)  # Prints a blank line then a divider to start Part 2
print("MISCLASSIFICATION ANALYSIS")  # Prints the heading for Part 2
print("=" * 55)  # Prints another divider

img_height = 224  # Sets image height to 224 pixels — must match what the models were trained on
img_width = 224   # Sets image width to 224 pixels
batch_size = 16   # Loads 16 images at a time

val_data_raw = tf.keras.utils.image_dataset_from_directory(
    "dataset/validation",  # Loads the validation images from the validation folder
    image_size=(img_height, img_width),  # Resizes every image to 224x224
    batch_size=batch_size,  # Loads 16 at a time
    shuffle=False  # Does not shuffle — keeps images in a fixed order so we can track which ones were wrong
)

class_names = val_data_raw.class_names  # Reads the folder names as class labels — high and low
print(f"\nClasses: {class_names}")  # Prints the class names to confirm they loaded correctly

all_images = []  # Creates an empty list to store all validation images
all_labels = []  # Creates an empty list to store all the correct labels

for images, labels in val_data_raw:  # Loops through every batch of validation images
    all_images.extend(images.numpy().astype(int))  # Converts images to numbers and adds them to the list
    all_labels.extend(labels.numpy().astype(int))  # Converts labels to integers and adds them to the list

all_images = np.array(all_images)  # Converts the image list into a NumPy array for easier handling
all_labels = np.array(all_labels)  # Converts the label list into a NumPy array
print(f"Validation images loaded: {len(all_images)}")  # Prints how many validation images were loaded — should be 40


def analyse_model(model, model_name, preprocessing="cnn"):  # Defines a reusable function that analyses any model — takes the model, its name, and which preprocessing method to use
    print(f"\n{'─'*50}")  # Prints a divider line
    print(f"Analysing: {model_name}")  # Prints which model is being analysed
    print(f"{'─'*50}")  # Prints another divider

    val_data = tf.keras.utils.image_dataset_from_directory(
        "dataset/validation",  # Reloads the validation images fresh for this model
        image_size=(img_height, img_width),  # Resizes to 224x224
        batch_size=batch_size,  # Loads 16 at a time
        shuffle=False  # Keeps fixed order so predictions match the correct images
    )

    if preprocessing == "mobilenet":
        val_data = val_data.map(lambda x, y: (
            tf.keras.applications.mobilenet_v2.preprocess_input(x), y  # Applies MobileNetV2's own specific preprocessing — scales pixels to between -1 and 1
        ))
    else:
        val_data = val_data.map(lambda x, y: (x / 255.0, y))  # For the CNN, divides pixels by 255 to normalise to 0-1

    preds = model.predict(val_data, verbose=0)  # Runs all 40 validation images through the model and gets a probability for each
    y_pred = (preds > 0.5).astype(int).flatten()  # Converts probabilities to 0 or 1 — above 0.5 is high severity, below is low severity

    correct_mask = (y_pred == all_labels)  # Creates a True/False list — True where the prediction matched the correct label
    incorrect_mask = (y_pred != all_labels)  # Creates a True/False list — True where the prediction was wrong

    print(f"✅ Correctly classified:  {correct_mask.sum()} / {len(all_labels)}")  # Prints how many images were classified correctly
    print(f"❌ Misclassified:         {incorrect_mask.sum()} / {len(all_labels)}")  # Prints how many images were classified wrongly

    print(f"\nMisclassified image details:")  # Prints a heading for the detailed breakdown
    print(f"{'Index':<8} {'True Label':<15} {'Predicted':<15} {'Confidence'}")  # Prints column headers
    print("─" * 50)  # Prints a divider under the headers

    misclassified_indices = np.where(incorrect_mask)[0]  # Gets the index numbers of every image that was misclassified

    for idx in misclassified_indices:  # Loops through each misclassified image
        true_label = class_names[all_labels[idx]]  # Gets the correct label for this image
        pred_label = class_names[y_pred[idx]]  # Gets what the model incorrectly predicted
        confidence = preds[idx][0]  # Gets the raw probability the model gave
        if y_pred[idx] == 1:
            conf_str = f"{confidence*100:.1f}% low"  # If predicted low, shows how confident the model was
        else:
            conf_str = f"{(1-confidence)*100:.1f}% high"  # If predicted high, shows the confidence the other way
        print(f"{idx:<8} {true_label:<15} {pred_label:<15} {conf_str}")  # Prints the details for this misclassified image

    if incorrect_mask.sum() == 0:
        print("✅ No misclassifications!")  # If the model got everything right, prints a success message and stops
        return

    n_incorrect = incorrect_mask.sum()  # Counts how many images were misclassified
    n_cols = 4  # Sets 4 images per row in the visual grid
    n_rows = int(np.ceil(n_incorrect / n_cols))  # Calculates how many rows are needed to fit all misclassified images

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))  # Creates a grid of empty image slots
    fig.suptitle(f"{model_name} — Misclassified Images ({n_incorrect} total)",
                 fontsize=14, fontweight='bold')  # Adds a title to the whole grid showing the model name and count

    if n_rows == 1:
        axes = axes.reshape(1, -1)  # If there's only one row, reshapes the axes so the rest of the code works correctly
    axes_flat = axes.flatten()  # Flattens the grid into a simple list so we can fill slots one by one

    for i, idx in enumerate(misclassified_indices):  # Loops through each misclassified image
        ax = axes_flat[i]  # Gets the next empty slot in the grid
        ax.imshow(all_images[idx])  # Displays the actual image in that slot
        true_label = class_names[all_labels[idx]]  # Gets the correct label
        pred_label = class_names[y_pred[idx]]  # Gets what the model predicted
        confidence = preds[idx][0]  # Gets the confidence probability
        if y_pred[idx] == 1:
            conf_pct = f"{confidence*100:.0f}%"  # Formats confidence as a percentage
        else:
            conf_pct = f"{(1-confidence)*100:.0f}%"  # Formats confidence the other way
        ax.set_title(
            f"Image #{idx}\nTrue: {true_label.upper()}\nPredicted: {pred_label.upper()} ({conf_pct})",
            fontsize=9, color='red'  # Adds a red title above each image showing its index, true label and wrong prediction
        )
        ax.axis('off')  # Removes the axis lines around each image to keep it clean

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')  # Hides any leftover empty grid slots so they don't show as blank boxes

    plt.tight_layout()  # Adjusts spacing so images and titles don't overlap
    filename = f"{model_name.lower().replace(' ', '_')}_misclassified.png"  # Creates a filename based on the model name e.g. cnn_model_misclassified.png
    plt.savefig(filename, dpi=150, bbox_inches='tight')  # Saves the grid of misclassified images as a high resolution image file
    plt.show()  # Displays the grid on screen
    print(f"\n✅ Saved: {filename}")  # Prints a confirmation that the file was saved


# ── BUILD AND TRAIN CNN ──────────────────────────────────────

print("\n" + "=" * 55)  # Prints a divider to separate this section
print("TRAINING CNN FOR ANALYSIS...")  # Prints a heading so it's clear the CNN is being trained
print("=" * 55)  # Prints another divider

train_cnn = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",  # Loads the training images from the train folder
    image_size=(img_height, img_width),  # Resizes to 224x224
    batch_size=batch_size,  # Loads 16 at a time
    shuffle=True  # Shuffles training images so the model doesn't learn the order
)
train_cnn = train_cnn.map(lambda x, y: (x / 255.0, y))  # Normalises training images by dividing pixels by 255

val_cnn = tf.keras.utils.image_dataset_from_directory(
    "dataset/validation",  # Loads validation images
    image_size=(img_height, img_width),  # Resizes to 224x224
    batch_size=batch_size,  # Loads 16 at a time
    shuffle=False  # No shuffle for validation
)
val_cnn = val_cnn.map(lambda x, y: (x / 255.0, y))  # Normalises validation images the same way

cnn = tf.keras.Sequential([  # Builds the CNN as a stack of layers
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),  # First scanning layer — 32 filters looking for basic features
    tf.keras.layers.MaxPooling2D(),  # Shrinks the image by keeping only the strongest features
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),  # Second scanning layer — 64 filters for more complex patterns
    tf.keras.layers.MaxPooling2D(),  # Shrinks again
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),  # Third scanning layer — 128 filters for even more complex features
    tf.keras.layers.MaxPooling2D(),  # Shrinks one final time
    tf.keras.layers.Flatten(),  # Converts 2D feature maps into a flat list of numbers
    tf.keras.layers.Dense(128, activation='relu'),  # Fully connected layer that combines all features to make a decision
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer — produces a probability between 0 and 1
])

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Sets up the CNN with Adam optimiser, binary crossentropy loss, and accuracy tracking
print("\nTraining CNN (10 epochs)...")  # Prints a message so you know training is starting
cnn.fit(train_cnn, validation_data=val_cnn, epochs=10, verbose=1)  # Trains the CNN for 10 epochs and shows progress each epoch

analyse_model(cnn, "CNN Model", preprocessing="cnn")  # Calls the analysis function on the trained CNN to find and display misclassified images


# ── BUILD AND TRAIN MOBILENETV2 ──────────────────────────────

print("\n" + "=" * 55)  # Prints a divider
print("TRAINING MOBILENETV2 FOR ANALYSIS...")  # Prints a heading for the MobileNetV2 section
print("=" * 55)  # Prints another divider

train_mob = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",  # Loads training images for MobileNetV2
    image_size=(img_height, img_width),  # Resizes to 224x224
    batch_size=batch_size,  # Loads 16 at a time
    shuffle=True  # Shuffles training images
)
train_mob = train_mob.map(
    lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y)  # Applies MobileNetV2's specific preprocessing — scales pixels to between -1 and 1 as required by the pretrained model
)

val_mob = tf.keras.utils.image_dataset_from_directory(
    "dataset/validation",  # Loads validation images for MobileNetV2
    image_size=(img_height, img_width),  # Resizes to 224x224
    batch_size=batch_size,  # Loads 16 at a time
    shuffle=False  # No shuffle for validation
)
val_mob = val_mob.map(
    lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y)  # Applies the same MobileNetV2 preprocessing to validation images
)

base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),  # Tells MobileNetV2 the size of images it will receive
    include_top=False,  # Removes MobileNetV2's original classification head — we don't need it since we're adding our own
    weights='imagenet'  # Loads the pretrained weights from ImageNet — this is the knowledge from 1.2 million images
)
base.trainable = False  # Freezes all the pretrained layers so they don't change during training — we only want to train our new layers

mob = tf.keras.Sequential([  # Builds the MobileNetV2 model by stacking layers
    base,  # The pretrained MobileNetV2 base — already knows how to read images
    tf.keras.layers.GlobalAveragePooling2D(),  # Converts the base output into a single flat vector by averaging — reduces dimensions
    tf.keras.layers.Dense(128, activation='relu'),  # Fully connected layer with 128 neurons to learn eczema-specific patterns
    tf.keras.layers.Dropout(0.3),  # Randomly switches off 30% of neurons during training to prevent overfitting
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer — produces a probability between 0 and 1 for high or low severity
])

mob.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Uses Adam with a smaller learning rate of 0.0001 — lower than the CNN because the pretrained weights are already good and we don't want to change them too aggressively
    loss='binary_crossentropy',  # Standard loss function for two-class problems
    metrics=['accuracy']  # Tracks accuracy during training
)

print("\nTraining MobileNetV2 (10 epochs)...")  # Prints a message so you know training is starting
mob.fit(train_mob, validation_data=val_mob, epochs=10, verbose=1)  # Trains MobileNetV2 for 10 epochs and shows progress

analyse_model(mob, "MobileNetV2 Model", preprocessing="mobilenet")  # Calls the analysis function on the trained MobileNetV2 to find and display misclassified images

print("\n" + "=" * 55)  # Prints a final divider
print("ANALYSIS COMPLETE")  # Prints a completion message
print("Files saved:")  # Lists the files that were saved
print("  - cnn_model_misclassified.png")  # The CNN misclassification grid image
print("  - mobilenetv2_model_misclassified.png")  # The MobileNetV2 misclassification grid image
print("=" * 55)  # Prints a final divider