import os
import shutil
import random
def copy2_file(source_file, destination_file):
    # Check if file already exists in the destination and skip copying if it does
    if os.path.exists(destination_file):
        print(f"File {destination_file} already exists. Skipping.")
        return
    
    shutil.copy2(source_file, destination_file)
'''
if __name__ == "__main__":
    source_folder_path = r"D:\AISHELL-3\train\wav"
    dest_folder_path = r"D:\AISHELL-3\train\My_mix_wav"
    bool_double_folder = True
    
    if bool_double_folder:
        total_path = []
        for dir in os.listdir(source_folder_path):
            dir_path = os.path.join(source_folder_path, dir)
            for filename in os.listdir(dir_path):
                source_file_path = os.path.join(dir_path, filename)
                total_path.append(source_file_path)
        # randommly distribute the wav files
        random.shuffle(total_path)
        # divide into 10 groups
        group_num = 10
        group_size = len(total_path) // group_num
        group_path = [total_path[i:i + group_size] for i in range(0, len(total_path), group_size)]
        small_dir_name = [f"wav_part_{i}" for i in range(1, len(group_path) + 1)]
        for i, paths in enumerate(group_path):
            small_dir_path = os.path.join(dest_folder_path, small_dir_name[i])
            os.makedirs(small_dir_path, exist_ok=True)
            for file_path in paths:
                destination_file_path = os.path.join(small_dir_path, file_path.split("\\")[-1])
                copy2_file(file_path, destination_file_path)
'''   
# prepare the audio dataset
# if __name__ == "__main__":
#     source_folder_path = r"D:\clone_audio\yang_chinesedata\fake\TTS_ELEVEN_SPEECH"
#     dest_folder_path1 = r"D:\clone_audio\chinese_audio_dataset_ver3\train_audio"
#     dest_folder_path2 = r"D:\clone_audio\chinese_audio_dataset_ver3\val_audio"
#     dest_folder_path3 = r"D:\clone_audio\chinese_audio_dataset_ver3\test_audio"
    
#     total_path = [i for i in os.listdir(source_folder_path)]
#     random.shuffle(total_path)
#     lst_n = [200, 60, 60]# train, val , test number
#     subset_path_train = total_path[0:lst_n[0]]
#     subset_path_val = total_path[lst_n[0]:lst_n[0]+lst_n[1]]
#     subset_path_test = total_path[lst_n[0]+lst_n[1]:lst_n[0]+lst_n[1]+lst_n[2]]

#     for filepath in subset_path_train:
#         source_file_path = os.path.join(source_folder_path, filepath)
#         destination_file_path = os.path.join(dest_folder_path1, filepath)
#         copy2_file(source_file_path, destination_file_path)   
#     for filepath in subset_path_val:
#         source_file_path = os.path.join(source_folder_path, filepath)
#         destination_file_path = os.path.join(dest_folder_path2, filepath)
#         copy2_file(source_file_path, destination_file_path)  
#     for filepath in subset_path_test:
#         source_file_path = os.path.join(source_folder_path, filepath)
#         destination_file_path = os.path.join(dest_folder_path3, filepath)
#         copy2_file(source_file_path, destination_file_path)  

# # put the augmented data into train_audio for suno bark
# if __name__ == "__main__":
#     source_folder_path = r"D:\clone_audio\chinese_audio_dataset_ver2_augmented_data\fake\suno_bark_audio_aug_timestretch"
#     dest_folder_path1 = r"D:\clone_audio\chinese_audio_dataset_ver3\temp"
#     dest_folder_path2 = r"D:\clone_audio\chinese_audio_dataset_ver3\train_audio_with20db_timestretch_suno_gl"
#     aug_path = os.listdir(dest_folder_path2)
#     random.shuffle(aug_path)
#     count = 0
#     for filename in aug_path:
#         if filename.startswith("fake"):
#             name = filename[:-4] + "_timestretch" + filename[-4:]
#             source_file_path = os.path.join(source_folder_path, name)
#             destination_file_path = os.path.join(dest_folder_path1, name)
#             copy2_file(source_file_path, destination_file_path)  
#             count += 1
#         if count >= 50:
#             break

# put the augmented data into train_audio for gl
if __name__ == "__main__":
    source_folder_path = r"D:\clone_audio\chinese_audio_dataset_ver2_augmented_data\fake\yang_fake\gl_audio_aug_timestretch"
    dest_folder_path1 = r"D:\clone_audio\chinese_audio_dataset_ver3\temp"
    dest_folder_path2 = r"D:\clone_audio\chinese_audio_dataset_ver3\train_audio_with20db_timestretch_suno_gl"
    aug_path = os.listdir(dest_folder_path2)
    random.shuffle(aug_path)
    count = 0
    for filename in aug_path:
        if filename.endswith("gl.wav"):
            name = filename[:-4] + "_timestretch" + filename[-4:]
            source_file_path = os.path.join(source_folder_path, name)
            destination_file_path = os.path.join(dest_folder_path1, name)
            copy2_file(source_file_path, destination_file_path)  
            count += 1
        if count >= 150:
            break

# if __name__ == "__main__":
#     source_folder_path = r"D:\clone_audio\lin_fake"
#     dest_folder_path1 = r"D:\clone_audio\lin_fake_rename"
#     # dest_folder_path2 = r"D:\clone_audio\chinese_audio_dataset_ver3\val_audio"
#     # dest_folder_path3 = r"D:\clone_audio\chinese_audio_dataset_ver3\test_audio"
#     # print(os.listdir(source_folder_path)[5])
#     for i in os.listdir(source_folder_path):
#         source_file_path = os.path.join(source_folder_path, i)
#         rename = i.split(" ")[0] +  "_" + i.split(" ")[1][1:-5] + i.split(" ")[1][-4:]
#         destination_file_path = os.path.join(dest_folder_path1, rename)
#         copy2_file(source_file_path, destination_file_path)  

# if __name__ == "__main__":
#     source_folder_path = r"D:\clone_audio\yang_chinesedata\fake\elevenlabs"
#     dest_folder_path1 = r"D:\clone_audio\yang_chinesedata\fake\TTS_ELEVEN_SPEECH"
#     # dest_folder_path2 = r"D:\clone_audio\chinese_audio_dataset_ver3\val_audio"
#     # dest_folder_path3 = r"D:\clone_audio\chinese_audio_dataset_ver3\test_audio"
#     # print(os.listdir(source_folder_path)[5])
#     for dir in os.listdir(source_folder_path):
#         inside_dir_path = os.path.join(source_folder_path, dir)
#         for filename in os.listdir(inside_dir_path):
#             source_file_path = os.path.join(inside_dir_path, filename)
#             destination_file_path = os.path.join(dest_folder_path1, filename)
#             copy2_file(source_file_path, destination_file_path)  
            
    