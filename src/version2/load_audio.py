import pandas as pd
import torchaudio

# info dataframe
# e.g
#       speaker_id      filename system_id class_name                                           filepath  target
# 0    LA_0079  LA_T_1138215         -   bonafide  D:/graduate_project/src/asvpoof-2019-dataset/L...       0
# 1    LA_0079  LA_T_1271820         -   bonafide  D:/graduate_project/src/asvpoof-2019-dataset/L...       0
train_df = pd.read_csv("train_info.csv")
print(train_df.head(5))


if __name__ == "__main__":
    ANNOTATIONS_FILE = "/home/valerio/datasets/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "/home/valerio/datasets/UrbanSound8K/audio"
    SAMPLE_RATE = 16000

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram,
                            SAMPLE_RATE)
    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]