# Model Experiment
#### Note:   
CH: https://hackmd.io/D5vDB1KiQM60tRxpeXZ9bg?both#731-%E8%A8%93%E7%B7%B4%E6%A8%A1%E5%9E%8B%E8%B3%87%E6%96%99%E7%B4%9A%E8%A8%AD%E8%A8%88%E5%A6%82%E4%B8%8A%E6%89%80%E7%A4%BA  
ENG: https://hackmd.io/@NsysuAnn/ryLvAgCF0
### 1. model_9_CH
**The original way: use mel-spectrogram as RGB pixel features.**
* input tensor : (batch size, 3*channels*, 128, 128)

### 2. model_10_CH
**Directly use mel-spectrogram data(128,216) as features.**
* input tensor : (batch size, 1*channels*, 128, 216)  
* Do normalization on each position across all batches
### 3. model_11_CH
**Use three features images(mel-spectrogram, mfcc, chroma) as RGB pixel features.**
* input tensor : (batch size, 9*channels*, 128, 128)

### 4. model_12_CH
**Use three features directly(mel-spectrogram, mfcc, chroma) to act as input of model12.**
* input tensor : (batch size, 3*channels*, 128, 216)
* 有用到插值法補齊mfcc跟chroma不夠的值
* Do normalization on each position of each channel across all batches

### 5. model_9_CH_ver2
**The original way: use mel-spectrogram as RGB pixel features.**
* input tensor : (batch size, 3*channels*, 128, 128)
* 在轉換成mel_spec時，沒有padding 0，讓他保持原始長度

### 6. model_9_CH_ver3
**The original way: use mel-spectrogram as RGB pixel features.**
* input tensor : (batch size, 3*channels*, 128, 128)
* 在轉換成mel_spec時，利用原始音檔來做padding，作法即為將音檔在複製在後面直到滿五秒

### 7. model_9_ENG
**The original way: use mel-spectrogram as RGB pixel features.**
* train on ASVspoof2019 dataset
* input tensor : (batch size, 3*channels*, 128, 128)
* 在轉換成mel_spec時，利用原始音檔來做padding，作法即為將音檔在複製在後面直到滿五秒

### 8. model_9_ENG_stacked_features
**The original way: use combined features image as RGB pixel features.**
* 將三種features合併起來成一張圖片
* train on ASVspoof2019 dataset
* input tensor : (batch size, 3*channels*, 128, 128)
* 在轉換成mel_spec時，利用原始音檔來做padding，作法即為將音檔在複製在後面直到滿五秒

### Testing code:
* `testing_model9_CH.py`, `testing_model9_CH_ver2.py`, `testing_model9_CH_ver3.py`, `testing_model9_ENG_stacked_features.py`, and `testing_model9_ENG.py` is associated with `SpectrogramDataset.py`.
* `testing_model10_CH.py` is associated with `AudioDataset.py`.
* `testing_model11_CH.py` is associated with `FeatureDataset.py`.
* `testing_model12_CH.py` is associated with `AudioDataset_ver2.py`.