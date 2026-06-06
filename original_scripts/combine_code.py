import os  # Imports the os library — used to check if a file exists on the computer

# List of all your Python files in the order you want them
files = [
    "test.py",               # First file to be added to the combined output
    "check_images.py",       # Second file to be added
    "fix_dataset_to_jpg.py", # Third file to be added
    "load_data.py",          # Fourth file to be added
    "cnn_model.py",          # Fifth file to be added
    "mobilenet_model.py",    # Sixth file to be added
    "dataset_analysis.py",   # Seventh file to be added
    "grayscale_experiment.py" # Eighth and final file to be added
]

# Output file name — change to your actual student ID and name
output_filename = "12345678-YourFirstName-YourLastName-MLModels.txt"  # The name of the final combined text file that will be created

with open(output_filename, "w") as outfile:  # Creates the output text file and opens it ready to write into — "w" means write mode, so it starts fresh
    for filename in files:  # Loops through each file name in the list above one by one
        if os.path.exists(filename):  # Checks if that file actually exists on the computer before trying to open it
            outfile.write("=" * 60 + "\n")  # Writes a line of 60 equal signs as a visual separator between files
            outfile.write(f"FILE: {filename}\n")  # Writes the name of the current file so it's clear which file is starting
            outfile.write("=" * 60 + "\n\n")  # Writes another line of equal signs to close the header, then adds a blank line
            with open(filename, "r") as infile:  # Opens the current Python file to read its contents — "r" means read mode
                outfile.write(infile.read())  # Reads everything inside that Python file and writes it into the combined output file
            outfile.write("\n\n")  # Adds two blank lines after each file to separate it from the next one
            print(f"Added: {filename}")  # Prints a confirmation message showing the file was successfully added
        else:
            print(f"NOT FOUND: {filename}")  # If the file doesn't exist, prints a warning instead of crashing

print(f"\nDone! Saved as: {output_filename}")  # Prints a final message when all files have been combined and saved