from pydub import AudioSegment
import os
import numpy as np
from audiomentations import Compose, TimeStretch
from observe_audio_function_ver2 import SAMPLE_RATE, load_audio
import librosa
import soundfile as sf

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

# Many folders in folder
# use for wav
if __name__ == "__main__":
    # some const variable
    augment_TimeStretch = Compose(
    [
        TimeStretch(min_rate=0.8, max_rate=1.25, leave_length_unchanged=False, p=1.0)
    ]
    # Minimum rate of change of total duration of the signal. A rate below 1 means the audio is slowed down.
    # Maximum rate of change of total duration of the signal. A rate greater than 1 means the audio is speed up.
    # don't keep the length unchanged
    ) 
    source_dir_path = r"D:\clone_audio\ASVspoof2019_MyDataset\dataset_ver1\audio\train_audio"
    dest_dir_path = r"D:\clone_audio\ASVspoof2019_MyDataset\dataset_ver1\audio\train_aduio_aug_timestretch"
    os.makedirs(dest_dir_path, exist_ok=True)
    for filename in os.listdir(source_dir_path):
        audio_path = os.path.join(source_dir_path, filename)
        #  load the original audio
        audio, sr = load_audio(audio_path, SAMPLE_RATE)

        # do TimeStretch
        augmented_audio = augment_TimeStretch(audio, sr)
                
        # Save augmented audio, preserving the original file extension
        output_path = os.path.join(dest_dir_path, f"{filename.split('.')[0]}_timestretch.{filename.split('.')[-1]}")
        # save_audio_with_pydub(augmented_audio, sr, output_path)
        sf.write(output_path, augmented_audio, sr)

    print(f"Finish creating new timestretch audioes into {dest_dir_path}.")
