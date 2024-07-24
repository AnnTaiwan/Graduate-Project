from pydub import AudioSegment
import os
import numpy as np
from audiomentations import Compose, AddGaussianNoise
from observe_audio_function_ver2 import SAMPLE_RATE, load_audio
import librosa
import soundfile as sf

def calculate_noise_std_from_snr(snr_db, rms_signal):
    """
    Calculate the standard deviation of noise required to achieve a given SNR in dB.

    Parameters:
    - snr_db: Signal-to-noise ratio in decibels.
    - rms_signal: RMS value of the signal.

    Returns:
    - noise_std: Standard deviation of the noise.
    """
    # Calculate the power ratio of the noise relative to the signal
    power_ratio = 10 ** (-snr_db / 10)
    
    # Calculate the standard deviation of the noise based on the power ratio
    noise_std = rms_signal * np.sqrt(power_ratio)
    
    return noise_std

def apply_noise_to_audio(audio, sr, snr_db=10):
    """
    Apply Gaussian noise to an audio signal to achieve a specific SNR.

    Parameters:
    - audio: The input audio signal.
    - sr: Sample rate of the audio.
    - snr_db: Desired signal-to-noise ratio in decibels.

    Returns:
    - augmented_audio: Audio signal with added Gaussian noise.
    """
    # Calculate the RMS value of the signal
    rms_signal = np.sqrt(np.mean(audio**2))
    
    # Calculate the required noise standard deviation for the desired SNR
    noise_std = calculate_noise_std_from_snr(snr_db, rms_signal)
    
    # Create an audio augmentation pipeline with Gaussian noise
    augment_add_loudness = Compose(
        [
            AddGaussianNoise(min_amplitude=noise_std, max_amplitude=noise_std, p=1)
        ]
    )
    
    # Apply the noise augmentation to the audio signal
    augmented_audio = augment_add_loudness(audio, sr)
    
    return augmented_audio

def load_audio_with_pydub(file_path, sample_rate=SAMPLE_RATE):
    """
    Load audio file using pydub, ensuring compatibility with various formats, and optionally resample to a given sample rate.

    Parameters:
    - file_path: Path to the audio file.
    - sample_rate: Desired sample rate for the audio. If None, use the original sample rate.

    Returns:
    - audio: Loaded audio signal.
    - sr: Sample rate of the audio.
    """
    audio_segment = AudioSegment.from_file(file_path)
    audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
    sr = audio_segment.frame_rate
    
    if audio_segment.channels == 2:
        audio = audio.reshape((-1, 2)).mean(axis=1)  # Convert to mono if stereo

    if sample_rate and sample_rate != sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate

    return audio, sr

def save_audio_with_pydub(audio, sr, file_path):
    """
    Save audio file using pydub to preserve the original format.

    Parameters:
    - audio: Audio signal to save.
    - sr: Sample rate of the audio.
    - file_path: Path to save the audio file.
    """
    audio_segment = AudioSegment(
        audio.tobytes(),
        frame_rate=sr,
        sample_width=audio.dtype.itemsize,
        channels=1
    )
    audio_segment.export(file_path, format=file_path.split('.')[-1])
'''
if __name__ == "__main__":
    # Setting SNR
    SNR_DB = 20
    source_dir_path = r"D:\clone_audio\yang_chinesedata\fake\elevenlabs"
    dest_dir_path = r"D:\clone_audio\chinese_audio_dataset_ver2\fake\yang_fake\elvenlabs_audio_aug_add_noise"
    os.makedirs(dest_dir_path, exist_ok=True)

    for dir_name in os.listdir(source_dir_path):
        inside_dir_path = os.path.join(source_dir_path, dir_name)
        
        if not os.path.isdir(inside_dir_path):
            continue  # Skip files that are not directories
        
        for filename in os.listdir(inside_dir_path):
            audio_path = os.path.join(inside_dir_path, filename)
            # Load the original audio
            audio, sr = load_audio_with_pydub(audio_path, SAMPLE_RATE)
                
            # Add white noise
            augmented_audio = apply_noise_to_audio(audio, sr, SNR_DB)
                
            # Save augmented audio, preserving the original file extension
            output_path = os.path.join(dest_dir_path, f"{filename.split('.')[0]}_noise{SNR_DB}db.{filename.split('.')[-1]}")
            save_audio_with_pydub(augmented_audio, sr, output_path)
        
        print(f"Finished creating new noise-added audios from {inside_dir_path} into {dest_dir_path}.")

    print(f"Finished creating new noise-added audios into {dest_dir_path}.")



'''
# Many folders in folder
# use for wav
if __name__ == "__main__":
    # some const variable
    # setting SNR_DB
    SNR_DB = 15
    source_dir_path = r"D:\clone_audio\yang_chinesedata\real\thchs30\thchs30"
    dest_dir_path = r"D:\clone_audio\chinese_audio_dataset_ver2\real\thchs30_audio_aug_timestretch"
    os.makedirs(dest_dir_path, exist_ok=True)
    for filename in os.listdir(source_dir_path):
        audio_path = os.path.join(source_dir_path, filename)
        #  load the original audio
        audio, sr = load_audio(audio_path, SAMPLE_RATE)

        # do adding white noise
        augmented_audio = apply_noise_to_audio(audio, sr, SNR_DB)
                
        # Save augmented audio, preserving the original file extension
        output_path = os.path.join(dest_dir_path, f"{filename.split('.')[0]}_timestretch.{filename.split('.')[-1]}")
        # save_audio_with_pydub(augmented_audio, sr, output_path)
        sf.write(output_path, augmented_audio, sr)

    print(f"Finish creating new adding-noise audioes into {dest_dir_path}.")
