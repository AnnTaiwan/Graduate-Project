from observe_audio_function_ver3 import load_audio,SAMPLE_RATE, AUDIO_LEN
import os
count = 0
source_dir = r"D:\clone_audio\chinese_audio_dataset_ver3\spoof\train_audio_with20db_timestretch_suno_gl"
for file in os.listdir(source_dir):
    path = os.path.join(source_dir, file)
    audio, sr = load_audio(path, SAMPLE_RATE)
    
    if len(audio) < AUDIO_LEN:
        count += 1

print(count, " / ", len(os.listdir(source_dir)))