'''
Use original audio to pad the audio to 5 seconds
Stack three features into one image
do min-max before stacking
'''
import matplotlib.pyplot as plt
import numpy as np
from observe_audio_function_ver3 import SAMPLE_RATE, AUDIO_LEN, load_audio, get_mel_spectrogram, get_mfcc, get_chroma
import os 
import torch.nn.functional as F
import torch

# Create a directory to store the spectrogram images
source_folder = r"D:\clone_audio\ASVspoof2019_MyDataset\dataset_ver1\audio\test_audio"
destination_folder = r"D:\clone_audio\ASVspoof2019_MyDataset\dataset_ver1\Image_noise20db_ver\test_Combined_Features_padding_original_audio"

if __name__ == "__main__":
    os.makedirs(destination_folder, exist_ok=True)
    print("Now, it's going to plot some images with three features(Mel-Spectrogram, MFCC, Chroma) stacked.")
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

        # Extract features
        spec = get_mel_spectrogram(audio)  
        mfcc = get_mfcc(audio)
        chroma = get_chroma(audio)

        # Resize MFCC and chroma to match the mel spectrogram size
        target_size = (128, 216)
        # `F.interpolate` expects shape like `(batch_size, num_channels, height, width)`
        mfcc_tensor = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 13, 216)
        chroma_tensor = torch.tensor(chroma).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 12, 216)

        mfcc_resized = F.interpolate(mfcc_tensor, size=target_size, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)  # Shape: (128, 216)
        chroma_resized = F.interpolate(chroma_tensor, size=target_size, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)  # Shape: (128, 216)

        # Convert mel spectrogram to tensor
        spec_tensor = torch.tensor(spec)

        # Normalize each feature to be in the range [0, 1] for visualization
        spec_norm = (spec_tensor - spec_tensor.min()) / (spec_tensor.max() - spec_tensor.min())
        mfcc_norm = (mfcc_resized - mfcc_resized.min()) / (mfcc_resized.max() - mfcc_resized.min())
        chroma_norm = (chroma_resized - chroma_resized.min()) / (chroma_resized.max() - chroma_resized.min())

        # Combine the features into a single image with 3 channels
        combined_image = torch.stack([spec_norm, mfcc_norm, chroma_norm], dim=0).numpy()  # Shape: (3, 128, 216)

        # Plot the combined features
        plt.imshow(combined_image.transpose(1, 2, 0), aspect='auto', origin='lower')
        plt.title('Combined Features (Mel-Spectrogram, MFCC, Chroma)')
        # Save the spectrogram image with a meaningful filename
        dest_filename = os.path.splitext(filename)[0] + "_Combined_Features" + ".png" #  `splitext` separate the subfilename and real name
        dest_filepath = os.path.join(destination_folder, dest_filename)
        plt.savefig(dest_filepath)
        # Close the figure to free up resources
        plt.close()

    print(f"Spectrogram images saved to {destination_folder}")
