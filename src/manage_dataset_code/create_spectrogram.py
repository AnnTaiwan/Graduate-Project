'''
For ASHELL-3 dataset
'''

import os
import matplotlib.pyplot as plt
from observe_audio_function import load_audio, get_mel_spectrogram, plot_mel_spectrogram, envelope, normalize_audio, SAMPLE_RATE, AUDIO_LEN
# get the constant and functions


# Create a directory to store the spectrogram images
destination_folder = r"D:\clone_audio\chinese_audio_spec\spoof_spec_test_suno_bark202_241"
source_folder = r"D:\clone_audio\temp" # totally, contains 174 folders

# for two folders
# if __name__ == "__main__":
#     os.makedirs(destination_folder, exist_ok=True)
#     for dir_name in os.listdir(source_folder): # root folder contains many folders, each containing many audioes
#         filepath_folder = os.path.join(source_folder, dir_name)
#         print("Now in ", dir_name)
#         for audio_name in os.listdir(filepath_folder):
#             audio_path = os.path.join(filepath_folder, audio_name)
#             audio, sr = load_audio(audio_path, sr=SAMPLE_RATE)
#             audio = audio[:AUDIO_LEN] # 由於採樣率為 16000 Hz，持續時間為 5 秒，因此音訊數據的總長度為 16000 * 5 = 80000 個樣本。
#             # delete the low energy part
#             mask, _ = envelope(audio, sr, threshold = 0.0005)  # can adjust the threshold
#             # audio = audio[mask]
#             spec = get_mel_spectrogram(audio)
#             # min-max the spec
#             # spec = normalize_audio(spec)
#             fig = plot_mel_spectrogram(spec)
#             plt.title("Spectrogram", fontsize=17)

#             # Save the spectrogram image with a meaningful filename
#             pic_name = f"spoof_{audio_name[:-4]}.png"  # Use single quotes inside the f-string
#             filepath = os.path.join(destination_folder, pic_name)
#             plt.savefig(filepath)

#             # Close the figure to free up resources
#             plt.close()

#     print(f"ALL Spectrogram images saved to {destination_folder}")

# for one folder
if __name__ == "__main__":
        os.makedirs(destination_folder, exist_ok=True)
        for audio_name in os.listdir(source_folder):
            audio_path = os.path.join(source_folder, audio_name)
            audio, sr = load_audio(audio_path, sr=SAMPLE_RATE)
            audio = audio[:AUDIO_LEN] # 由於採樣率為 16000 Hz，持續時間為 5 秒，因此音訊數據的總長度為 16000 * 5 = 80000 個樣本。
            # delete the low energy part
            mask, _ = envelope(audio, sr, threshold = 0.0005)  # can adjust the threshold
            # audio = audio[mask]
            spec = get_mel_spectrogram(audio)
            # min-max the spec
            # spec = normalize_audio(spec)
            fig = plot_mel_spectrogram(spec)
            plt.title("Spectrogram", fontsize=17)

            # Save the spectrogram image with a meaningful filename
            pic_name = f"spoof_suno_bark{audio_name[:-4]}.png"  # Use single quotes inside the f-string
            filepath = os.path.join(destination_folder, pic_name)
            plt.savefig(filepath)

            # Close the figure to free up resources
            plt.close()

        print(f"ALL Spectrogram images saved to {destination_folder}")
