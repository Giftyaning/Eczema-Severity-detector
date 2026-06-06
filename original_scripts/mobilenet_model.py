import tensorflow as tf  # Imports TensorFlow — the main library used to build and train the MobileNetV2 model
import numpy as np  # Imports NumPy — used for number calculations like converting predictions to 0s and 1s
import matplotlib.pyplot as plt  # Imports Matplotlib — used to draw and save the accuracy and loss graphs
import seaborn as sns  # Imports Seaborn — used to draw the confusion matrix as a colour heatmap
from sklearn.metrics import confusion_matrix, classification_report  # Imports tools to measure how well the model performed

# ─────────────────────────────────────────
# 1. SETTINGS
# ─────────────────────────────────────────

img_height = 224  # Sets every image height to 224 pixels — MobileNetV2 requires this exact size
img_width = 224   # Sets every image width to 224 pixels
batch_size = 16   # Loads and processes 16 images at a time to save memory

# ─────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────

train_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",                      # Tells TensorFlow where the training images are stored
    image_size=(img_height, img_width),   # Resizes every training image to 224x224 when loading
    batch_size=batch_size,                # Loads 16 images at a time
    shuffle=True                          # Randomly shuffles training images each epoch so the model doesn't learn the order
)

val_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/validation",                 # Tells TensorFlow where the validation images are stored
    image_size=(img_height, img_width),   # Resizes every validation image to 224x224
    batch_size=batch_size,                # Loads 16 at a time
    shuffle=False                         # Does not shuffle — keeps a fixed order so predictions match the correct images
)

class_names = train_data.class_names  # Reads the subfolder names as class labels — high and low
print("Classes:", class_names)  # Prints the class names to confirm they loaded correctly

# ─────────────────────────────────────────
# 3. PREPROCESS FOR MOBILENETV2
# ─────────────────────────────────────────

# MobileNetV2 needs pixels between -1 and 1 (not 0-1)
# So we use its own preprocessing function

def preprocess(image, label):  # Defines a reusable function that preprocesses any image and keeps its label unchanged
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # Applies MobileNetV2's specific preprocessing — scales pixel values from 0-255 down to between -1 and 1 as required by the pretrained model
    return image, label  # Returns the preprocessed image and its label

train_data = train_data.map(preprocess)  # Applies the preprocessing function to every training image
val_data = val_data.map(preprocess)  # Applies the same preprocessing to every validation image

# ─────────────────────────────────────────
# 4. LOAD MOBILENETV2 BASE (pretrained)
# ─────────────────────────────────────────

# include_top=False means we remove the original classification head
# We will add our own head for eczema classification
# weights='imagenet' means it already knows about 1000 types of images

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),  # Tells MobileNetV2 the size and channels of images it will receive — 224x224 with 3 colour channels
    include_top=False,          # Removes MobileNetV2's original classification head — it was trained for 1000 ImageNet classes which we don't need
    weights='imagenet'          # Loads the pretrained weights from ImageNet — this is the knowledge learned from 1.2 million images
)

# Freeze the base model
# This means we do NOT change what MobileNetV2 already learned
# We only train our new layers on top

base_model.trainable = False  # Freezes all pretrained layers so their weights don't change during training — only our new layers will be trained

# ─────────────────────────────────────────
# 5. BUILD FULL MODEL
# ─────────────────────────────────────────

model = tf.keras.Sequential([  # Builds the full model as a stack of layers
    base_model,                                      # The frozen pretrained MobileNetV2 base — already knows how to read images
    tf.keras.layers.GlobalAveragePooling2D(),        # Takes the output from the base and averages it into a single flat vector — reduces the data down to a manageable size before the decision layers
    tf.keras.layers.Dense(128, activation='relu'),   # Fully connected layer with 128 neurons — learns eczema specific patterns from the features MobileNetV2 extracted. relu turns any negative value to 0 for faster learning
    tf.keras.layers.Dropout(0.3),                    # Randomly switches off 30% of neurons during each training step to prevent the model from overfitting
    tf.keras.layers.Dense(1, activation='sigmoid')   # Output layer — produces a single probability between 0 and 1. Above 0.5 is high severity, below 0.5 is low severity
])

model.summary()  # Prints a summary of every layer and the total number of parameters — only 164,097 new parameters were trained

# ─────────────────────────────────────────
# 6. COMPILE MODEL
# ─────────────────────────────────────────

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Uses Adam optimiser with a small learning rate of 0.0001 — kept low so the pretrained weights are not overwritten too aggressively
    loss='binary_crossentropy',                                 # Standard loss function for two class problems — measures how wrong the predictions are
    metrics=['accuracy']                                        # Tracks accuracy during training so we can monitor improvement each epoch
)

# ─────────────────────────────────────────
# 7. TRAIN MODEL
# ─────────────────────────────────────────

history = model.fit(
    train_data,                  # Feeds the preprocessed training images into the model
    validation_data=val_data,    # Tests on validation images after each epoch to check for overfitting
    epochs=10                    # Trains for 10 rounds — the model sees all training images 10 times
)

# ─────────────────────────────────────────
# 8. ACCURACY & LOSS GRAPHS
# ─────────────────────────────────────────

plt.figure(figsize=(12, 4))  # Creates a blank figure 12 units wide and 4 units tall to hold both graphs

# Accuracy graph
plt.subplot(1, 2, 1)  # Places the next graph in position 1 of a 1-row 2-column layout
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o', color='blue')    # Draws the training accuracy line in blue with a dot at each epoch
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o', color='orange') # Draws the validation accuracy line in orange — if this is close to blue the model is learning well
plt.title('MobileNetV2 - Accuracy')  # Adds a title to the accuracy graph
plt.xlabel('Epoch')                   # Labels the x-axis — each point is one training round
plt.ylabel('Accuracy')                # Labels the y-axis
plt.legend()                          # Adds a legend showing which line is training and which is validation
plt.grid(True)                        # Adds a background grid for easier reading

# Loss graph
plt.subplot(1, 2, 2)  # Places the next graph in position 2
plt.plot(history.history['loss'], label='Train Loss', marker='o', color='blue')    # Draws the training loss line — should go down as the model improves
plt.plot(history.history['val_loss'], label='Val Loss', marker='o', color='orange') # Draws the validation loss line — if this moves with training loss the model is generalising well
plt.title('MobileNetV2 - Loss')  # Adds a title to the loss graph
plt.xlabel('Epoch')               # Labels the x-axis
plt.ylabel('Loss')                # Labels the y-axis
plt.legend()                      # Adds the legend
plt.grid(True)                    # Adds the grid

plt.tight_layout()  # Adjusts spacing so both graphs don't overlap each other
plt.savefig('mobilenet_accuracy_loss.png', dpi=150)  # Saves the graphs as a high resolution image file
plt.show()  # Displays the graphs on screen
print("✅ Graph saved: mobilenet_accuracy_loss.png")  # Prints a confirmation message

# ─────────────────────────────────────────
# 9. CONFUSION MATRIX + CLASSIFICATION REPORT
# ─────────────────────────────────────────

y_pred_probs = model.predict(val_data)  # Runs all 40 validation images through the model and gets a probability for each one
y_pred = (y_pred_probs > 0.5).astype(int).flatten()  # Converts probabilities to 0 or 1 — above 0.5 becomes 1 (high severity), below becomes 0 (low severity)

y_true = np.concatenate([y for x, y in val_data], axis=0).astype(int).flatten()  # Collects all the correct labels from the validation set so we can compare against predictions

cm = confusion_matrix(y_true, y_pred)  # Compares true labels against predicted labels and builds the confusion matrix

plt.figure(figsize=(6, 5))  # Creates a blank figure for the confusion matrix
sns.heatmap(
    cm,                          # The confusion matrix data to visualise
    annot=True,                  # Shows the actual numbers inside each box of the heatmap
    fmt='d',                     # Formats the numbers as whole integers not decimals
    cmap='Greens',               # Colours the heatmap in shades of green — darker means higher numbers
    xticklabels=class_names,     # Labels the columns with class names — high and low
    yticklabels=class_names      # Labels the rows with class names — high and low
)
plt.title('MobileNetV2 - Confusion Matrix')  # Adds a title to the confusion matrix
plt.xlabel('Predicted Label')                # Labels the x-axis — what the model predicted
plt.ylabel('True Label')                     # Labels the y-axis — what the correct answer actually was
plt.tight_layout()                           # Adjusts spacing so nothing is cut off
plt.savefig('mobilenet_confusion_matrix.png', dpi=150)  # Saves the confusion matrix as a high resolution image file
plt.show()  # Displays the confusion matrix on screen
print("✅ Confusion matrix saved: mobilenet_confusion_matrix.png")  # Prints a confirmation message

print("\n── Classification Report ──────────────────")  # Prints a header line before the report
print(classification_report(y_true, y_pred, target_names=class_names))  # Prints precision, recall, F1-score and accuracy for each class — full breakdown of model performance