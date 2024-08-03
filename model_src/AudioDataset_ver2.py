'''
Here, can obtain three features directly
'''
from observe_audio_function_ver3 import SAMPLE_RATE, AUDIO_LEN, get_mel_spectrogram, load_audio, get_mfcc, get_chroma, N_MELS, SPEC_WIDTH
import numpy as np
import torch
import torch.nn.functional as F

class AudioDataset_ver2(torch.utils.data.Dataset):
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
        # Extract features
        spec = get_mel_spectrogram(audio)  # Shape: (128, 216)
        mfcc = get_mfcc(audio)  # Shape: (13, 216)
        chroma = get_chroma(audio)  # Shape: (12, 216)

        # Convert MFCC and chroma to tensor and add a batch and channel dimension
        mfcc_tensor = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 13, 216)
        chroma_tensor = torch.tensor(chroma).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 12, 216)

        # EACH DIMENSION'S TARGET SIZE
        target_size = (N_MELS, SPEC_WIDTH)
        # Interpolate to target size (128, 216)
        mfcc_resized = F.interpolate(mfcc_tensor, size=target_size, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)  # Shape: (128, 216)
        chroma_resized = F.interpolate(chroma_tensor, size=target_size, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)  # Shape: (128, 216)

        # Convert mel spectrogram to tensor if not already
        spec_tensor = torch.tensor(spec)

        # Combine the features into a single 3-channel feature vector
        combined_feature = torch.stack([spec_tensor, mfcc_resized, chroma_resized], dim=0)  # Shape: (3, 128, 216)
        combined_feature = self._normalize(combined_feature) # do the normalization, return shape torch.Size([3, 128, 216])
        label = np.array(self.labels[idx])
        return combined_feature, label           # return 模型訓練所需的資訊

    def _normalize(self, spec):
        # spec's shape: (3, 128, 216)
        # mean's shape: (1, 3, 128, 216)
        # Despite the shape is different, PyTorch’s broadcasting mechanism is alike with Numpy's.
        (spec - self.mean.squeeze(0)) / self.std.squeeze(0)
        return spec