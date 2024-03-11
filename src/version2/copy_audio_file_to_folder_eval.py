import os
import pandas as pd
import random
import shutil

# Get each 2000 files in each folders. 

# Create a directory to store the audio
destination_folder1 = "D:/graduate_project/src/LAEval_audio_shuffle1"
destination_folder2 = "D:/graduate_project/src/LAEval_audio_shuffle2"
destination_folder3 = "D:/graduate_project/src/LAEval_audio_shuffle3"
source_folder = "D:/graduate_project/src/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_eval/flac"

NUM = 2000 # number of files
if __name__ == "__main__":
    os.makedirs(destination_folder1, exist_ok=True)
    os.makedirs(destination_folder2, exist_ok=True)
    os.makedirs(destination_folder3, exist_ok=True)

    train_df = pd.read_csv(r"D:\graduate_project\src\version2\eval_info.csv")
    # 0~2547 is 0, 2548~24843 is 1.
    L = list(range(0, 71237)) # get the first 500 audios index
    random.shuffle(L)
    l1 = L[:NUM]
    l2 = L[NUM:2*NUM]
    l3 = L[2*NUM: 3*NUM]

    row_bonafide1 = train_df.iloc[l1].copy()
    row_bonafide2 = train_df.iloc[l2].copy()
    row_bonafide3 = train_df.iloc[l3].copy()


    # Iterate through files in the source folder
    for index, r in row_bonafide1.iterrows():
        source_path = os.path.join(source_folder, r["filename"]+".flac")
        destination_path = os.path.join(destination_folder1, r["filename"]+".flac")

        # Copy the file to the destination folder instead of moving
        shutil.copy(source_path, destination_path)
    
    print(f"{NUM} files move successfully into {destination_folder1}.")
    ###############
    # Iterate through files in the source folder
    for index, r in row_bonafide2.iterrows():
        source_path = os.path.join(source_folder, r["filename"]+".flac")
        destination_path = os.path.join(destination_folder2, r["filename"]+".flac")

        # Copy the file to the destination folder instead of moving
        shutil.copy(source_path, destination_path)
    
    print(f"{NUM} files move successfully into {destination_folder2}.")
    #################
     # Iterate through files in the source folder
    for index, r in row_bonafide3.iterrows():
        source_path = os.path.join(source_folder, r["filename"]+".flac")
        destination_path = os.path.join(destination_folder3, r["filename"]+".flac")

        # Copy the file to the destination folder instead of moving
        shutil.copy(source_path, destination_path)
    
    print(f"{NUM} files move successfully into {destination_folder3}.")
    #################
