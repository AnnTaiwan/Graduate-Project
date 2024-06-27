'''
Separate the bonafide CH spec file into different folders.
'''
import os
import random
import shutil

def separate_folder_into_parts(source_dir, destination_base_dir, num_parts=10):
    # List all files in the source directory
    all_files = os.listdir(source_dir)
    files = [f for f in all_files if os.path.isfile(os.path.join(source_dir, f))]
    
    # Shuffle the list of files
    random.shuffle(files)
    
    # Calculate the number of files per part
    num_files = len(files)
    part_size = num_files // num_parts
    remainder = num_files % num_parts
    
    # Create subfolders and distribute files
    file_index = 0
    for part in range(1, num_parts + 1):
        part_dir = os.path.join(destination_base_dir, f'part_{part}')
        os.makedirs(part_dir, exist_ok=True)
        
        # Calculate the number of files for this part
        if part <= remainder:
            current_part_size = part_size + 1
        else:
            current_part_size = part_size
        
        # Copy files to the part directory
        for _ in range(current_part_size):
            if file_index < num_files:
                source_file = os.path.join(source_dir, files[file_index])
                destination_file = os.path.join(part_dir, files[file_index])
                shutil.copy2(source_file, destination_file)
                file_index += 1

# Specify the source and destination directories
source_directory = r"D:\clone_audio\chinese_audio_spec\bonafide_spec"
destination_base_directory = r"D:\clone_audio\chinese_audio_spec\bonafide_spec_base"

# Call the function to separate the folder into parts
separate_folder_into_parts(source_directory, destination_base_directory, num_parts=10)
