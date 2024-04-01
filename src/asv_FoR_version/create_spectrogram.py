import os
import matplotlib.pyplot as plt
from observe_audio_function import load_audio, get_mel_spectrogram, plot_mel_spectrogram, envelope, normalize_audio, SAMPLE_RATE, AUDIO_LEN
# get the constant and functions


# Create a directory to store the spectrogram images
destination_folder = r"D:\FoR\FoR_for_norm\for-norm\training\spec\spec_real"
source_folder = r"D:\FoR\FoR_for_norm\for-norm\training\real"
# Save every 3000 pictures to prevent shutting down
if __name__ == "__main__":
    os.makedirs(destination_folder, exist_ok=True)
    file_list = os.listdir(source_folder)
    for filename in file_list[17000//2:17000]:
        filepath2 = os.path.join(source_folder, filename)
        audio, sr = load_audio(filepath2, sr=SAMPLE_RATE)
        audio = audio[:AUDIO_LEN]
        # delete the low energy part
        mask, _ = envelope(audio, sr, threshold = 0.0005)  # can adjust the threshold
        # audio = audio[mask]
        spec = get_mel_spectrogram(audio)
        # min-max the spec
        # spec = normalize_audio(spec)
        fig = plot_mel_spectrogram(spec)
        plt.title("Spectrogram", fontsize=17)

        # Save the spectrogram image with a meaningful filename
        filename = f"spec_fake_{filename.split('.')[0]}.png"  # Use single quotes inside the f-string
        filepath = os.path.join(destination_folder, filename)
        plt.savefig(filepath)

        # Close the figure to free up resources
        plt.close()

    print(f"Spectrogram images saved to {destination_folder}")
