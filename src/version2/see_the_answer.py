'''
see the images' true labels.
'''
import os
import pandas as pd

dev_df = pd.read_csv("train_info.csv")
folder_path = r"C:\Users\User\Desktop\IMAGES"
# List files in the folder
file_names = os.listdir(folder_path)

# Filter only files (excluding directories)
file_names = [f for f in file_names if os.path.isfile(os.path.join(folder_path, f))]

# Iterate through the file names and find matching rows in the DataFrame
for file_name in file_names:
    matching_row = dev_df[dev_df['filename'] == file_name[5:-4]]
    if not matching_row.empty:
        print(f"File: {file_name}")
        print(matching_row["target"])


