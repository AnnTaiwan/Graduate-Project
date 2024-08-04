import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from observe_audio_function_ver3 import SAMPLE_RATE, AUDIO_LEN, load_audio, get_mel_spectrogram, plot_mel_spectrogram
import os 
# Create a directory to store the spectrogram images
source_folder = r"D:\clone_audio\chinese_audio_dataset_ver3\bonafide\test_audio"
destination_folder = r"D:\clone_audio\chinese_audio_dataset_ver3\test_mel_spec_without_padding"

if __name__ == "__main__":
    os.makedirs(destination_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        filepath = os.path.join(source_folder, filename)
        audio, sr = load_audio(filepath, sr=SAMPLE_RATE)
        # cut the audio
        audio = audio[:AUDIO_LEN]

        spec = get_mel_spectrogram(audio)
        fig = plot_mel_spectrogram(spec)
        plt.title("Mel-Spectrogram", fontsize=17)

        # Save the spectrogram image with a meaningful filename
        dest_filename = f"bonafide_melspec_{filename[:-4]}.png"  # Use single quotes inside the f-string
        # dest_filename = f"spoof_melspec_{filename[:-4]}.png"  # Use single quotes inside the f-string
        dest_filepath = os.path.join(destination_folder, dest_filename)
        plt.savefig(dest_filepath)
        # Close the figure to free up resources
        plt.close()

    print(f"Spectrogram images saved to {destination_folder}")