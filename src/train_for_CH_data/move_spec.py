'''
move spectrogram into another folder
'''
import os
import random
import shutil

def move_random_files(source_dir, destination_dir, num_files):
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)
    
    # List all files in the source directory
    all_files = os.listdir(source_dir)
    files = [f for f in all_files if os.path.isfile(os.path.join(source_dir, f))]
    
    # Check if there are enough files to copy
    if len(files) < num_files:
        raise ValueError(f"Not enough files in the source directory. Found {len(files)}, but need {num_files}.")
    
    # Randomly select num_files files
    selected_files = random.sample(files, num_files)
    
    # Copy the selected files to the destination directory
    for file_name in selected_files:
        source_file = os.path.join(source_dir, file_name)
        destination_file = os.path.join(destination_dir, file_name)
        
        # Check if file already exists in the destination and skip copying if it does
        if os.path.exists(destination_file):
            print(f"File {destination_file} already exists. Skipping.")
            continue
        
        shutil.move(source_file, destination_file)

# Specify the source and destination directories and number of files to copy
source_directory = r"D:\clone_audio\chinese_audio_spec\spoof_spec"
destination_directory = r"D:\clone_audio\chinese_audio_spec\spoof_spec_1"
number_of_files_to_copy = 100

# Call the function to move the files
move_random_files(source_directory, destination_directory, number_of_files_to_copy)
