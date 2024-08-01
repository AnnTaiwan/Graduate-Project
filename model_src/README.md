# Model Experiment
Note: https://hackmd.io/D5vDB1KiQM60tRxpeXZ9bg?both#731-%E8%A8%93%E7%B7%B4%E6%A8%A1%E5%9E%8B%E8%B3%87%E6%96%99%E7%B4%9A%E8%A8%AD%E8%A8%88%E5%A6%82%E4%B8%8A%E6%89%80%E7%A4%BA

### 1. model_9_CH
**The original way: use mel-spectrogram as RGB pixel features.**
* input tensor : (batch size, 3*channels*, 128, 128)

### 2. model_10_CH
**Directly use mel-spectrogram data(128,216) as features.**
* input tensor : (batch size, 1*channels*, 128, 216)

### 3. model_11_CH
**Use three features images(mel-spectrogram, mfcc, chroma) as RGB pixel features.**
* input tensor : (batch size, 9*channels*, 128, 128)

### Testing code:
* `testing_model9_CH.py` is associated with `SpectrogramDataset.py`.
* `testing_model10_CH.py` is associated with `AudioDataset.py`.
* `testing_model11_CH.py` is associated with `FeatureDataset.py`.