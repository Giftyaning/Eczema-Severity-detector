## To scan dataset for broken / corrupted images
from PIL import Image ## Imports the pillow library to open and verify image files.
import os ## Imports the os library used to navigate through folders and build files paths.

dataset_path = "dataset" ## Defines the root folder path where all the images are stored.

for root, dirs, files in os.walk(dataset_path):
    for file in files: ## Loops through every all the files un the current folder.
        file_path = os.path.join(root, file) ## Builds the file path by joining the folder path and filename.
        try:
            with Image.open(file_path) as img: ## Attempts to open the image file using pillow library.
                img.verify() ## Checks the file structure to check it's validity.
        except Exception as e: ## To prevent any errors from crashing the script.
            print(f"Bad file: {file_path} -> {e}") ## To print the full path of the bad file.