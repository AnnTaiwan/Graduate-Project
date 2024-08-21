import os
import random
import shutil

def copy_random_files(source_dir, destination_dir, num_files=50):
    # 確保目標目錄存在
    os.makedirs(destination_dir, exist_ok=True)

    # 獲取來源目錄中的所有文件
    all_files = os.listdir(source_dir)

    # 根據文件名的前綴過濾出 'spoof' 和 'bonafide' 文件
    spoof_files = [f for f in all_files if f.startswith("spoof")]
    bonafide_files = [f for f in all_files if f.startswith("bonafide")]

    # 確保有足夠的文件可供複製
    if len(spoof_files) < num_files:
        raise ValueError(f"Not enough spoof files. Found {len(spoof_files)}, required {num_files}.")
    if len(bonafide_files) < num_files:
        raise ValueError(f"Not enough bonafide files. Found {len(bonafide_files)}, required {num_files}.")

    # 隨機選擇指定數量的 'spoof' 和 'bonafide' 文件
    selected_spoof_files = random.sample(spoof_files, num_files)
    selected_bonafide_files = random.sample(bonafide_files, num_files)

    # 將選擇的文件複製到目標目錄
    for file_name in selected_spoof_files + selected_bonafide_files:
        source_file = os.path.join(source_dir, file_name)
        destination_file = os.path.join(destination_dir, file_name)

        # 檢查文件是否已經存在於目標目錄，避免覆蓋
        if os.path.exists(destination_file):
            print(f"File {destination_file} already exists. Skipping.")
            continue
        
        shutil.copy2(source_file, destination_file)
        # print(f"Copied {file_name} to {destination_dir}")
    print("OK")

if __name__ == '__main__':
    # 指定來源和目標目錄
    source_directory = r"D:\clone_audio\chinese_audio_dataset_ver3\test_mel_spec_padding_original_audio"
    destination_directory = r"D:\clone_audio\chinese_audio_dataset_ver3\used_for_kv260_testing\test_CH_100_3"
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    # 執行函數，從 source_directory 中複製 50 個 'bonafide' 和 50 個 'spoof' 文件到 destination_directory
    copy_random_files(source_directory, destination_directory, num_files=50)
