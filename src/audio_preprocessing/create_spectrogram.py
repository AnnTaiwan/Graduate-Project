import warnings

# Suppress specific UserWarning from numba
warnings.filterwarnings("ignore", message="FNV hashing is not implemented in Numba")

import os
import matplotlib.pyplot as plt
from observe_audio_function import load_audio, get_mel_spectrogram, plot_mel_spectrogram, envelope, normalize_audio, SAMPLE_RATE, AUDIO_LEN, denoise
import numpy as np

# Create a directory to store the spectrogram images
destination_folder = "spec"
source_folder = "audio"
segment_duration = 5  # Segment duration in seconds
threshold = 0.0005  # threshold for envelope

if __name__ == "__main__":
    os.makedirs(destination_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        filepath2 = os.path.join(source_folder, filename)
        audio, sr = load_audio(filepath2, sr=SAMPLE_RATE)
        
        rn_audio = denoise(audio, sr=SAMPLE_RATE)
        # delete margin in the beginning and end
        mask, y_mean = envelope(rn_audio, sr, threshold)
        # find the first and last point larger than threshold
        start_idx = np.argmax(mask)
        end_idx = len(mask) - np.argmax(mask[::-1])
        audio_trimmed = rn_audio[start_idx:end_idx]
        
        print("start_idx: ", start_idx, "end_idx: ", end_idx, "Actual END:", len(mask))
        # Split audio into segments of 5 seconds
        num_segments = int(len(audio_trimmed) / (segment_duration * sr))
        # if audio length less than 5 sec.
        if num_segments == 0:
          num_segments += 1
        
        for i in range(num_segments):
            start_sample = i * segment_duration * sr
            end_sample = min(start_sample + segment_duration * sr, len(audio_trimmed))
            audio_segment = audio_trimmed[start_sample:end_sample]
            
            
            
            # Generate the mel spectrogram
            spec = get_mel_spectrogram(audio_segment)
            
            # Plot the mel spectrogram
            fig = plot_mel_spectrogram(spec)
            plt.title(f"Spectrogram Segment {i+1}", fontsize=17)
            
            # Save the spectrogram image with a meaningful filename
            segment_filename = f"spec_{filename[:-4]}_segment_{i+1}.png"
            save_filepath = os.path.join(destination_folder, segment_filename)
            plt.savefig(save_filepath)
            
            # Close the figure to free up resources
            plt.close()

    print(f"Spectrogram images saved to {destination_folder}")

'''
# used for not cutting audio
if __name__ == "__main__":
    os.makedirs(destination_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        filepath2 = os.path.join(source_folder, filename)
        audio, sr = load_audio(filepath2, sr=SAMPLE_RATE)
        audio = audio[:AUDIO_LEN]
        rn_audio = denoise(audio, sr=SAMPLE_RATE)
        # delete the low energy part
        # mask, _ = envelope(rn_audio, sr, threshold = 0.0005)  # can adjust the threshold
        # audio = audio[mask]
        spec = get_mel_spectrogram(rn_audio)
        # min-max the spec
        # spec = normalize_audio(spec)
        fig = plot_mel_spectrogram(spec)
        plt.title("Spectrogram", fontsize=17)

        # Save the spectrogram image with a meaningful filename
        filename = f"spec_{filename[:-4]}.png"  # Use single quotes inside the f-string
        filepath = os.path.join(destination_folder, filename)
        plt.savefig(filepath)

        # Close the figure to free up resources
        plt.close()

    print(f"Spectrogram images saved to {destination_folder}")
'''