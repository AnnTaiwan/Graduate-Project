'''
Copy some random spec_img into a new folder.
'''

import os
import shutil
import pandas as pd
import random
# List of filenames
filenames = [
    "spec_LA_E_2453205.png",
    "spec_LA_E_6421288.png",
    "spec_LA_E_6049858.png",
    "spec_LA_E_7926975.png",
    "spec_LA_E_3698225.png",
    "spec_LA_E_2131508.png",
    "spec_LA_E_3661306.png",
    "spec_LA_E_4325150.png",
    "spec_LA_E_6873356.png",
    "spec_LA_E_8938637.png",
    "spec_LA_E_9715267.png",
    "spec_LA_E_5555053.png",
    "spec_LA_E_7091396.png",
    "spec_LA_E_4636126.png",
    "spec_LA_E_2553134.png",
    "spec_LA_E_7620083.png",
    "spec_LA_E_3324859.png",
    "spec_LA_E_5184440.png",
    "spec_LA_E_2839569.png",
    "spec_LA_E_9604092.png",
    "spec_LA_E_8816152.png",
    "spec_LA_E_5549172.png",
    "spec_LA_E_7445982.png",
    "spec_LA_E_1774628.png",
    "spec_LA_E_3862064.png",
    "spec_LA_E_2997838.png",
    "spec_LA_E_6847028.png",
    "spec_LA_E_6193124.png",
    "spec_LA_E_1886954.png"
]

# # Source directory (replace with your actual source directory)
# source_dir = r"D:\graduate_project\src\spec_LAEval_audio_shuffle1_NOT_preprocessing"

# # Destination directory (replace with your desired destination directory)
# dest_dir = r"C:\Users\User\Desktop\eval1_1"

# # Create the destination directory if it doesn't exist
# os.makedirs(dest_dir, exist_ok=True)

# # Copy each file to the destination directory
# for filename in filenames:
#     source_path = os.path.join(source_dir, filename)
#     dest_path = os.path.join(dest_dir, filename)
    
#     # Check if the source file exists before copying
#     if os.path.exists(source_path):
#         shutil.copy(source_path, dest_path)
#         print(f"Copied {filename} to {dest_path}")
#     else:
#         print(f"File {filename} does not exist in the source directory.")

# print("All files have been copied.")

# Source directory (replace with your actual source directory)
source_dir = r"D:\graduate_project\src\spec_LAEval_audio_shuffle1_NOT_preprocessing"

# Destination directory (replace with your desired destination directory)
dest_dir = r"C:\Users\User\Desktop\eval100_5"
train_df = pd.read_csv("eval_info.csv")

image_paths = [os.path.join(source_dir, filename) for filename in os.listdir(source_dir)]


    # Load Labels
    # Assuming 'train_df' has columns 'filename' and 'target'
    # skip the "spec_" and ".png"
random.shuffle(image_paths)
bonafide_file_name = [os.path.basename(path) for path in image_paths[:100]]
# label = [(os.path.basename(path)[5:-4], train_df[train_df["filename"] == os.path.basename(path)[5:-4]]["target"].values[0]) for path in image_paths[:20]]
# print(label)
# print(bonafide_file_name[:20])
filenames = bonafide_file_name[:100]
# Create the destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Copy each file to the destination directory
for filename in filenames:
    source_path = os.path.join(source_dir, filename)
    dest_path = os.path.join(dest_dir, filename)
    
    # Check if the source file exists before copying
    if os.path.exists(source_path):
        shutil.copy(source_path, dest_path)
        print(f"Copied {filename} to {dest_path}")
    else:
        print(f"File {filename} does not exist in the source directory.")

print("All files have been copied.")