import os
import matplotlib.pyplot as plt
import pandas as pd
import random
import shutil

# Get the bonafide and spoof each 500 files. 

# Create a directory to store the audio
destination_folder = "D:/graduate_project/src/LATrain_audio_shuffle1"
source_folder = "D:/graduate_project/src/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_train/flac"

NUM = 500 # number of files
if __name__ == "__main__":
    os.makedirs(destination_folder, exist_ok=True)

    train_df = pd.read_csv("train_info.csv")
    # 0~2579 is 0, 2580~25379 is 1.
    l1 = list(range(0, 2580)) # get the first 500 audios index
    random.shuffle(l1)
    l1 = l1[:NUM]
    row_bonafide = train_df.iloc[l1].copy()


    l2 = list(range(2580, 25380))
    random.shuffle(l2)
    l2 = l2[:NUM]
    row_spoof = train_df.iloc[l2].copy()

    # Iterate through files in the source folder
    for index, r in row_bonafide.iterrows():
        source_path = os.path.join(source_folder, r["filename"]+".flac")
        destination_path = os.path.join(destination_folder, r["filename"]+".flac")

        # Copy the file to the destination folder instead of moving
        shutil.copy(source_path, destination_path)
    
    print(f"{NUM} Bonafide files move successfully into {destination_folder}.")
    # Iterate through files in the source folder
    for index, r in row_spoof.iterrows():
        source_path = os.path.join(source_folder, r["filename"]+".flac")
        destination_path = os.path.join(destination_folder, r["filename"]+".flac")

        # Copy the file to the destination folder instead of moving
        shutil.copy(source_path, destination_path)

    print(f"{NUM} Spoof files move successfully into {destination_folder}.")
