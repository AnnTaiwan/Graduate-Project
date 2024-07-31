'''
The other method:
Turn into an image by using `opencv`, then save this image in dimension (128, 128).
'''
import librosa
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 假設 audio 是已加載的音頻數據
audio, sr = librosa.load('t_feature_extraction_output_9.wav', sr=22050)

# 設置參數
N_FFT = 2048
HOP_LEN = 512
N_MELS = 128
N_MFCC = 128

# 提取特徵
mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN, n_mels=N_MELS)
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LEN)
chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN)

# 將特徵轉換為 dB (適用於 mel spectrogram)
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
print(mel_spectrogram_db.shape)
# 定義將特徵矩陣轉換為圖像的函數
def feature_to_image(feature, size=(128, 128)):
    # 調整大小
    feature_resized = cv2.resize(feature, size, interpolation=cv2.INTER_AREA)
    # 歸一化到0-255之間
    feature_normalized = cv2.normalize(feature_resized, None, 0, 255, cv2.NORM_MINMAX)
    # 將其轉換為無符號8位整數
    feature_image = feature_normalized.astype(np.uint8)
    return feature_image

# 將特徵轉換為圖像
mel_image = feature_to_image(mel_spectrogram_db)
mfcc_image = feature_to_image(mfcc)
chroma_image = feature_to_image(chroma)

# 保存圖像
cv2.imwrite('mel_spectrogram.jpg', mel_image)
cv2.imwrite('mfcc.jpg', mfcc_image)
cv2.imwrite('chroma.jpg', chroma_image)

# 顯示圖像
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(mel_image, cmap='hot')
plt.title('Mel Spectrogram')

plt.subplot(1, 3, 2)
plt.imshow(mfcc_image, cmap='hot')
plt.title('MFCC')

plt.subplot(1, 3, 3)
plt.imshow(chroma_image, cmap='hot')
plt.title('Chroma')

plt.show()
