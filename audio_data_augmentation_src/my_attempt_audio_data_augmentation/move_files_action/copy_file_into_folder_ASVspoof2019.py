'''
Used for ASVspoof2019 dataset
'''
import os
import shutil
import pandas as pd
import random
def copy2_file(source_file, destination_file):
    # Check if file already exists in the destination and skip copying if it does
    if os.path.exists(destination_file):
        print(f"File {destination_file} already exists. Skipping.")
        return
    
    shutil.copy2(source_file, destination_file)
'''
if __name__ == "__main__":
    source_dir = r"D:\graduate_project\src\asvpoof-2019-dataset\LA\LA\ASVspoof2019_LA_dev\flac"
    dest_dir = r"D:\clone_audio\ASVspoof2019_MyDataset\dataset_ver1\audio\val_audio"

    df = pd.read_csv("train_info.csv")

    file_list = os.listdir(source_dir)
    random.shuffle(file_list)
    num_spoof = 0
    num_bonafide = 0

    NUM = 2000 # each amount of audio for one label
    for filename in file_list:
        if df[df["filename"] == filename[:-5]]["target"].values[0] == 0 and num_bonafide < NUM:
            new_filename = "bonafide_" + filename # create a new name
            source_filepath = os.path.join(source_dir, filename)
            dest_filepath = os.path.join(dest_dir, new_filename)
            copy2_file(source_filepath, dest_filepath)
            num_bonafide += 1
        elif df[df["filename"] == filename[:-5]]["target"].values[0] == 1 and num_spoof < NUM:
            new_filename = "spoof_" + filename # create a new name
            source_filepath = os.path.join(source_dir, filename)
            dest_filepath = os.path.join(dest_dir, new_filename)
            copy2_file(source_filepath, dest_filepath)
            num_spoof += 1

        # check if there are enough files.
        if num_spoof >= NUM and num_bonafide >= NUM:
            break
'''
'''
if __name__ == "__main__":
    source_dir = r"D:\graduate_project\src\asvpoof-2019-dataset\LA\LA\ASVspoof2019_LA_dev\flac"
    dest_dir = r"D:\clone_audio\ASVspoof2019_MyDataset\dataset_ver1\audio\val_audio"

    df = pd.read_csv("dev_info.csv")

    file_list = os.listdir(source_dir)
    random.shuffle(file_list)
    num_spoof = 0
    num_bonafide = 0

    NUM = 1000  # each amount of audio for one label
    for filename in file_list:
        query_result = df[df["filename"] == filename[:-5]]
        if not query_result.empty: # due to some audio files aren't recorded into csv file
            target_value = query_result["target"].values[0]
            if target_value == 0 and num_bonafide < NUM:
                new_filename = "bonafide_" + filename  # create a new name
                source_filepath = os.path.join(source_dir, filename)
                dest_filepath = os.path.join(dest_dir, new_filename)
                copy2_file(source_filepath, dest_filepath)
                num_bonafide += 1
            elif target_value == 1 and num_spoof < NUM:
                new_filename = "spoof_" + filename  # create a new name
                source_filepath = os.path.join(source_dir, filename)
                dest_filepath = os.path.join(dest_dir, new_filename)
                copy2_file(source_filepath, dest_filepath)
                num_spoof += 1

            # check if there are enough files.
            if num_spoof >= NUM and num_bonafide >= NUM:
                break
        else:
            print(f"No target information found for filename: {filename[:-5]}")

    print(f"Finished copying files: {num_bonafide} bonafide, {num_spoof} spoof")
'''
if __name__ == "__main__":
    source_dir = r"D:\graduate_project\src\asvpoof-2019-dataset\LA\LA\ASVspoof2019_LA_eval\flac"
    dest_dir = r"D:\clone_audio\ASVspoof2019_MyDataset\dataset_ver1\audio\test_audio"

    df = pd.read_csv("eval_info.csv")

    file_list = os.listdir(source_dir)
    random.shuffle(file_list)
    num_spoof = 0
    num_bonafide = 0

    NUM = 1000  # each amount of audio for one label
    for filename in file_list:
        query_result = df[df["filename"] == filename[:-5]]
        if not query_result.empty: # due to some audio files aren't recorded into csv file
            target_value = query_result["target"].values[0]
            if target_value == 0 and num_bonafide < NUM:
                new_filename = "bonafide_" + filename  # create a new name
                source_filepath = os.path.join(source_dir, filename)
                dest_filepath = os.path.join(dest_dir, new_filename)
                copy2_file(source_filepath, dest_filepath)
                num_bonafide += 1
            elif target_value == 1 and num_spoof < NUM:
                new_filename = "spoof_" + filename  # create a new name
                source_filepath = os.path.join(source_dir, filename)
                dest_filepath = os.path.join(dest_dir, new_filename)
                copy2_file(source_filepath, dest_filepath)
                num_spoof += 1

            # check if there are enough files.
            if num_spoof >= NUM and num_bonafide >= NUM:
                break
        else:
            print(f"No target information found for filename: {filename[:-5]}")

    print(f"Finished copying files: {num_bonafide} bonafide, {num_spoof} spoof")