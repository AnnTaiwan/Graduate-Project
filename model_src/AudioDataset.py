from observe_audio_function_ver3 import SAMPLE_RATE, AUDIO_LEN, get_mel_spectrogram, load_audio
import numpy as np
import torch

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, labels, mean, std):
        self.filenames = filenames    # 資料集的所有檔名
        self.labels = labels          # 影像的標籤
        self.mean = mean              # train dataset's mean
        self.std = std                # train dataset's std

    def __len__(self):
        return len(self.filenames)    # return DataSet 長度

    def __getitem__(self, idx):       # idx: Index of filenames
        # Loading audio
        audio, _ = load_audio(self.filenames[idx], SAMPLE_RATE)
        # padding 0
        if len(audio) < AUDIO_LEN:
            audio = np.pad(audio, (0, AUDIO_LEN - len(audio)), 'constant') # padding zero
        else:
            audio = audio[:AUDIO_LEN]
        spec = get_mel_spectrogram(audio)
        spec = self._normalize(spec) # do the normalization
        spec = np.expand_dims(spec, axis=0)
        label = np.array(self.labels[idx])
        return spec, label           # return 模型訓練所需的資訊

    def _normalize(self, spec):
        (spec - self.mean) / self.std
        return spec