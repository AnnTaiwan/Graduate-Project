import matplotlib.pyplot as plt
import numpy as np
from observe_audio_function_ver3 import SAMPLE_RATE, AUDIO_LEN, load_audio, get_mfcc, plot_mfcc
import os 
# Create a directory to store the spectrogram images
source_folder = r"D:\clone_audio\chinese_audio_dataset_ver3\spoof\test_audio"
destination_folder = r"D:\clone_audio\chinese_audio_dataset_ver3\test_mfcc"

if __name__ == "__main__":
    os.makedirs(destination_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        filepath = os.path.join(source_folder, filename)
        audio, sr = load_audio(filepath, sr=SAMPLE_RATE)
        # pad the audio or cut the audio
        if len(audio) < AUDIO_LEN:
            audio = np.pad(audio, (0, AUDIO_LEN - len(audio)), 'constant') # padding zero
        else:
            audio = audio[:AUDIO_LEN]

        mfcc = get_mfcc(audio)
        fig = plot_mfcc(mfcc)
        plt.title("MFCC", fontsize=17)

        # Save the spectrogram image with a meaningful filename
        # dest_filename = f"bonafide_mfcc_{filename[:-4]}.png"  # Use single quotes inside the f-string
        dest_filename = f"spoof_mfcc_{filename[:-4]}.png"  # Use single quotes inside the f-string
        dest_filepath = os.path.join(destination_folder, dest_filename)
        plt.savefig(dest_filepath)
        # Close the figure to free up resources
        plt.close()

    print(f"Spectrogram images saved to {destination_folder}")