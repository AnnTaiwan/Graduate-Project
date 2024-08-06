'''
Use original audio to pad the audio to 5 seconds
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from observe_audio_function_ver3 import SAMPLE_RATE, AUDIO_LEN, load_audio, get_mel_spectrogram, plot_mel_spectrogram
import os 
# Create a directory to store the spectrogram images
source_folder = r"D:\clone_audio\ASVspoof2019_MyDataset\dataset_ver1\audio\test_audio"
destination_folder = r"D:\clone_audio\ASVspoof2019_MyDataset\dataset_ver1\Image_noise20db_ver\test_mel_spec_padding_original_audio"

if __name__ == "__main__":
    os.makedirs(destination_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        filepath = os.path.join(source_folder, filename)
        audio, sr = load_audio(filepath, sr=SAMPLE_RATE)
        # pad the audio with the original audio or cut the audio
        if len(audio) < AUDIO_LEN:
            length_audio = len(audio)
            repeat_count = (AUDIO_LEN + length_audio - 1) // length_audio  # Calculate the `ceiling` of AUDIO_LEN / length_audio
            audio = np.tile(audio, repeat_count)[:AUDIO_LEN]  # Repeat and cut to the required length
        else:
            audio = audio[:AUDIO_LEN]

        spec = get_mel_spectrogram(audio)
        fig = plot_mel_spectrogram(spec)
        plt.title("Mel-Spectrogram", fontsize=17)

        # Save the spectrogram image with a meaningful filename
        dest_filename = os.path.splitext(filename)[0] + "_mel_spec" + ".png" #  `splitext` separate the subfilename and real name
        dest_filepath = os.path.join(destination_folder, dest_filename)
        plt.savefig(dest_filepath)
        # Close the figure to free up resources
        plt.close()

    print(f"Spectrogram images saved to {destination_folder}")