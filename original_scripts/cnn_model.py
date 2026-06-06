## To build and train the CNN model

import tensorflow as tf  # Imports TensorFlow — the main library used to build and train the model
import numpy as np  # Imports NumPy — used for number calculations like converting predictions to 0s and 1s
import matplotlib.pyplot as plt  # Imports Matplotlib — used to draw and save the accuracy and loss graphs
import seaborn as sns  # Imports Seaborn — used to draw the confusion matrix as a colour heatmap
from sklearn.metrics import confusion_matrix, classification_report  # Imports tools to measure how well the model performed

## To set image sizes - 224 is standard input size
img_height = 224  # Sets every image height to 224 pixels before feeding into the model
img_width = 224   # Sets every image width to 224 pixels — must match the height for a square input
batch_size = 16   # Trains 16 images at a time instead of all at once — saves memory, standard.

## To load data
train_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",  # Tells TensorFlow where the training images are stored
    image_size=(img_height, img_width),  # Resizes every image to 224x224 when loading
    batch_size=batch_size,  # Loads 16 images at a time
    shuffle=True  # Randomly mixes up the training images each epoch so the model doesn't learn the order
)

val_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/validation",  # Tells TensorFlow where the validation images are stored
    image_size=(img_height, img_width),  # Resizes validation images to 224x224 as well
    batch_size=batch_size,  # Loads 16 validation images at a time
    shuffle=False  # Does NOT shuffle validation images — order doesn't matter here, we just want consistent results
)

class_names = train_data.class_names  # Reads the folder names as class labels — in this case 'high' and 'low'
print("Classes:", class_names)  # Prints the class names to confirm they loaded correctly

## Normalise data
normalization_layer = tf.keras.layers.Rescaling(1./255)  # Creates a layer that divides every pixel value by 255 — converts pixel range from 0-255 down to 0-1, which helps the model learn faster

train_data = train_data.map(lambda x, y: (normalization_layer(x), y))  # Applies the normalisation to every training image, keeping the labels (y) unchanged
val_data = val_data.map(lambda x, y: (normalization_layer(x), y))  # Applies the same normalisation to every validation image

## Build CNN model
model = tf.keras.Sequential([  # Creates the model as a straight sequence of layers stacked on top of each other

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),  # First scanning layer — uses 32 filters, each looking at a 3x3 pixel area at a time, detecting basic features like edges. relu means any negative value becomes 0 to help the model learn faster.
    tf.keras.layers.MaxPooling2D(),  # Shrinks the image by keeping only the strongest feature in each 2x2 area — reduces size and keeps important information

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # Second scanning layer — uses 64 filters to detect more complex patterns like shapes and textures
    tf.keras.layers.MaxPooling2D(),  # Shrinks the image again — same as before

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),  # Third scanning layer — uses 128 filters to detect even more complex features
    tf.keras.layers.MaxPooling2D(),  # Shrinks the image one final time

    tf.keras.layers.Flatten(),  # Converts the 2D feature maps into a single long list of numbers so it can be fed into the decision layers
    tf.keras.layers.Dense(128, activation='relu'),  # A fully connected layer with 128 neurons — combines all the features detected above to start making a decision
    tf.keras.layers.Dense(1, activation='sigmoid')  # The output layer — produces a single number between 0 and 1. Above 0.5 = high severity, below 0.5 = low severity
])

model.summary()  # Prints a summary of the model showing every layer and the total number of parameters — 11 million in this case

## Compile model
model.compile(
    optimizer='adam',  # Uses the Adam optimiser — automatically adjusts how fast the model learns during training
    loss='binary_crossentropy',  # The loss function for two-class problems — measures how wrong the model's predictions are
    metrics=['accuracy']  # Tells TensorFlow to track accuracy during training so we can see it improve each epoch
)

## Train model
history = model.fit(
    train_data,  # Feeds the training images into the model
    validation_data=val_data,  # Also tests on validation images after each epoch to check for overfitting
    epochs=10  # Trains for 10 rounds — the model sees all training images 10 times
)

## Accuracy & loss graphs
plt.figure(figsize=(12, 4))  # Creates a blank figure 12 units wide and 4 units tall to hold both graphs

## Accuracy graph
plt.subplot(1, 2, 1)  # Places the next graph in position 1 of a 1-row, 2-column layout
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o', color='blue')  # Draws the training accuracy line in blue with a dot at each epoch
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o', color='orange')  # Draws the validation accuracy line in orange — if this is much lower than blue, the model is overfitting
plt.title('CNN Model - Accuracy')  # Adds a title to the accuracy graph
plt.xlabel('Epoch')  # Labels the x-axis as Epoch — each point represents one training round
plt.ylabel('Accuracy')  # Labels the y-axis as Accuracy
plt.legend()  # Adds a legend showing which line is training and which is validation
plt.grid(True)  # Adds a background grid to make the graph easier to read

## Loss graph
plt.subplot(1, 2, 2)  # Places the next graph in position 2 of the same layout
plt.plot(history.history['loss'], label='Train Loss', marker='o', color='blue')  # Draws the training loss line — should go down as the model improves
plt.plot(history.history['val_loss'], label='Val Loss', marker='o', color='orange')  # Draws the validation loss — if this goes up while training loss goes down, the model is overfitting
plt.title('CNN Model - Loss')  # Adds a title to the loss graph
plt.xlabel('Epoch')  # Labels the x-axis
plt.ylabel('Loss')  # Labels the y-axis
plt.legend()  # Adds the legend
plt.grid(True)  # Adds the grid

plt.tight_layout()  # Automatically adjusts spacing so the two graphs don't overlap
plt.savefig('cnn_accuracy_loss.png', dpi=150)  # Saves the graph as an image file at high resolution
plt.show()  # Displays the graph on screen
print("✅ Graph saved: cnn_accuracy_loss.png")  # Prints a confirmation message

## Confusion matrix + classification report
y_pred_probs = model.predict(val_data)  # Runs all 40 validation images through the model and gets a probability for each one
y_pred = (y_pred_probs > 0.5).astype(int).flatten()  # Converts probabilities to 0 or 1 — anything above 0.5 becomes 1 (high severity), anything below becomes 0 (low severity)

y_true = np.concatenate([y for x, y in val_data], axis=0).astype(int).flatten()  # Collects the actual correct labels from the validation set so we can compare them against predictions

cm = confusion_matrix(y_true, y_pred)  # Compares the true labels against predicted labels and builds the confusion matrix — shows correct and incorrect predictions

plt.figure(figsize=(6, 5))  # Creates a blank figure to draw the confusion matrix on
sns.heatmap(
    cm,  # The confusion matrix data to visualise
    annot=True,  # Shows the actual numbers inside each box of the heatmap
    fmt='d',  # Formats the numbers as whole integers, not decimals
    cmap='Blues',  # Colours the heatmap in shades of blue — darker means higher numbers
    xticklabels=class_names,  # Labels the columns with class names — high and low
    yticklabels=class_names  # Labels the rows with class names — high and low
)
plt.title('CNN - Confusion Matrix')  # Adds a title to the confusion matrix
plt.xlabel('Predicted Label')  # Labels the x-axis — what the model predicted
plt.ylabel('True Label')  # Labels the y-axis — what the correct answer actually was
plt.tight_layout()  # Adjusts spacing so nothing is cut off
plt.savefig('cnn_confusion_matrix.png', dpi=150)  # Saves the confusion matrix as an image file
plt.show()  # Displays the confusion matrix on screen
print("✅ Confusion matrix saved: cnn_confusion_matrix.png")  # Prints a confirmation message

## Classification report
print("\n── Classification Report ──────────────────")  # Prints a header line before the report
print(classification_report(y_true, y_pred, target_names=class_names))  # Prints precision, recall, F1-score and accuracy for each class — gives a full breakdown of model performance