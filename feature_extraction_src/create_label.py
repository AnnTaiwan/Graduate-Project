import os
import pandas as pd

dict1 = {
    "File_name": [],
    "type": [],
    "Mel_spec_path": [],
    "MFCC_path": [],
    "Chroma_path": [],
    "Label": []
}

if __name__ == "__main__":
    source_folder_path_train = r"D:\clone_audio\chinese_audio_dataset_ver3\bonafide\train_audio_with20db_timestretch_thchs30"
    source_folder_path_val = r"D:\clone_audio\chinese_audio_dataset_ver3\bonafide\val_audio"
    source_folder_path_test = r"D:\clone_audio\chinese_audio_dataset_ver3\bonafide\test_audio"

    # Train data
    dict1["File_name"].extend([filename[:-4] for filename in os.listdir(source_folder_path_train)])
    dict1["type"].extend(["train" for _ in range(len(os.listdir(source_folder_path_train)))])
    dict1["Mel_spec_path"].extend([fr"D:\clone_audio\chinese_audio_dataset_ver3\train_mel_spec\bonafide_melspec_{filename[:-4]}.png" for filename in os.listdir(source_folder_path_train)])
    dict1["MFCC_path"].extend([fr"D:\clone_audio\chinese_audio_dataset_ver3\train_mfcc\bonafide_mfcc_{filename[:-4]}.png" for filename in os.listdir(source_folder_path_train)])
    dict1["Chroma_path"].extend([fr"D:\clone_audio\chinese_audio_dataset_ver3\train_chroma\bonafide_chroma_{filename[:-4]}.png" for filename in os.listdir(source_folder_path_train)])
    dict1["Label"].extend([0 for _ in range(len(os.listdir(source_folder_path_train)))])

    # Validation data
    dict1["File_name"].extend([filename[:-4] for filename in os.listdir(source_folder_path_val)])
    dict1["type"].extend(["val" for _ in range(len(os.listdir(source_folder_path_val)))])
    dict1["Mel_spec_path"].extend([fr"D:\clone_audio\chinese_audio_dataset_ver3\val_mel_spec\bonafide_melspec_{filename[:-4]}.png" for filename in os.listdir(source_folder_path_val)])
    dict1["MFCC_path"].extend([fr"D:\clone_audio\chinese_audio_dataset_ver3\val_mfcc\bonafide_mfcc_{filename[:-4]}.png" for filename in os.listdir(source_folder_path_val)])
    dict1["Chroma_path"].extend([fr"D:\clone_audio\chinese_audio_dataset_ver3\val_chroma\bonafide_chroma_{filename[:-4]}.png" for filename in os.listdir(source_folder_path_val)])
    dict1["Label"].extend([0 for _ in range(len(os.listdir(source_folder_path_val)))])

    # Test data
    dict1["File_name"].extend([filename[:-4] for filename in os.listdir(source_folder_path_test)])
    dict1["type"].extend(["test" for _ in range(len(os.listdir(source_folder_path_test)))])
    dict1["Mel_spec_path"].extend([fr"D:\clone_audio\chinese_audio_dataset_ver3\test_mel_spec\bonafide_melspec_{filename[:-4]}.png" for filename in os.listdir(source_folder_path_test)])
    dict1["MFCC_path"].extend([fr"D:\clone_audio\chinese_audio_dataset_ver3\test_mfcc\bonafide_mfcc_{filename[:-4]}.png" for filename in os.listdir(source_folder_path_test)])
    dict1["Chroma_path"].extend([fr"D:\clone_audio\chinese_audio_dataset_ver3\test_chroma\bonafide_chroma_{filename[:-4]}.png" for filename in os.listdir(source_folder_path_test)])
    dict1["Label"].extend([0 for _ in range(len(os.listdir(source_folder_path_test)))])

    spoof_source_folder_path_train = r"D:\clone_audio\chinese_audio_dataset_ver3\spoof\train_audio_with20db_timestretch_suno_gl"
    spoof_source_folder_path_val = r"D:\clone_audio\chinese_audio_dataset_ver3\spoof\val_audio"
    spoof_source_folder_path_test = r"D:\clone_audio\chinese_audio_dataset_ver3\spoof\test_audio"

    # Train data (spoof)
    dict1["File_name"].extend([filename[:-4] for filename in os.listdir(spoof_source_folder_path_train)])
    dict1["type"].extend(["train" for _ in range(len(os.listdir(spoof_source_folder_path_train)))])
    dict1["Mel_spec_path"].extend([fr"D:\clone_audio\chinese_audio_dataset_ver3\train_mel_spec\spoof_melspec_{filename[:-4]}.png" for filename in os.listdir(spoof_source_folder_path_train)])
    dict1["MFCC_path"].extend([fr"D:\clone_audio\chinese_audio_dataset_ver3\train_mfcc\spoof_mfcc_{filename[:-4]}.png" for filename in os.listdir(spoof_source_folder_path_train)])
    dict1["Chroma_path"].extend([fr"D:\clone_audio\chinese_audio_dataset_ver3\train_chroma\spoof_chroma_{filename[:-4]}.png" for filename in os.listdir(spoof_source_folder_path_train)])
    dict1["Label"].extend([1 for _ in range(len(os.listdir(spoof_source_folder_path_train)))])

    # Validation data (spoof)
    dict1["File_name"].extend([filename[:-4] for filename in os.listdir(spoof_source_folder_path_val)])
    dict1["type"].extend(["val" for _ in range(len(os.listdir(spoof_source_folder_path_val)))])
    dict1["Mel_spec_path"].extend([fr"D:\clone_audio\chinese_audio_dataset_ver3\val_mel_spec\spoof_melspec_{filename[:-4]}.png" for filename in os.listdir(spoof_source_folder_path_val)])
    dict1["MFCC_path"].extend([fr"D:\clone_audio\chinese_audio_dataset_ver3\val_mfcc\spoof_mfcc_{filename[:-4]}.png" for filename in os.listdir(spoof_source_folder_path_val)])
    dict1["Chroma_path"].extend([fr"D:\clone_audio\chinese_audio_dataset_ver3\val_chroma\spoof_chroma_{filename[:-4]}.png" for filename in os.listdir(spoof_source_folder_path_val)])
    dict1["Label"].extend([1 for _ in range(len(os.listdir(spoof_source_folder_path_val)))])

    # Test data (spoof)
    dict1["File_name"].extend([filename[:-4] for filename in os.listdir(spoof_source_folder_path_test)])
    dict1["type"].extend(["test" for _ in range(len(os.listdir(spoof_source_folder_path_test)))])
    dict1["Mel_spec_path"].extend([fr"D:\clone_audio\chinese_audio_dataset_ver3\test_mel_spec\spoof_melspec_{filename[:-4]}.png" for filename in os.listdir(spoof_source_folder_path_test)])
    dict1["MFCC_path"].extend([fr"D:\clone_audio\chinese_audio_dataset_ver3\test_mfcc\spoof_mfcc_{filename[:-4]}.png" for filename in os.listdir(spoof_source_folder_path_test)])
    dict1["Chroma_path"].extend([fr"D:\clone_audio\chinese_audio_dataset_ver3\test_chroma\spoof_chroma_{filename[:-4]}.png" for filename in os.listdir(spoof_source_folder_path_test)])
    dict1["Label"].extend([1 for _ in range(len(os.listdir(spoof_source_folder_path_test)))])

    output_df = pd.DataFrame(dict1)
    csv_file_path = r"../model_src/chinese_audio_dataset_ver3_info.csv"
    output_df.to_csv(csv_file_path, index=False)
    print(f"Data has been successfully exported to {csv_file_path}")
