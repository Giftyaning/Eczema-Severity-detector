## To convert all images to jpg format
from PIL import Image
import os

dataset_path = "dataset"

##Defines all image file extensions that the script will attempt to convert - to avoid it converting other files.
supported_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        file_path = os.path.join(root, file) ## Builds the full path to the current files.

        if not file.lower().endswith(supported_extensions):
            continue ## Skips files not recognised image formats.

        try:
            with Image.open(file_path) as img: ## Opens the image file using pillow.
                img = img.convert("RGB") ## Convert the image to RGB colour mode 
                new_file_path = os.path.splitext(file_path)[0] + ".jpg" ## Builds the new file path with a .jpg extension
                img.save(new_file_path, "JPEG") ## Save the converted image as a JPEG file at the new file path

            if file_path != new_file_path:
                os.remove(file_path) ## Deletes the original non-JPEG file

            print(f"Fixed: {file_path} -> {new_file_path}") ## Prints a confirmation message showing the original file path and the new converted file path

        except Exception as e: ## To catch any errors that occurs.
            print(f"Skipped: {file_path} -> {e}") ## Prints the file path and the error message