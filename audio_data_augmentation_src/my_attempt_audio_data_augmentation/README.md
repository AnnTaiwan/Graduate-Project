# This directory is used for doing **audio data augmentation**.
### Right now, it can only apply to .wav files because `pydub` can't save mp3 files well now.
#### So, I only apply below methods to fake(suno_bark, gl) and real(thchs30) datasets.
### All methods can be implmented by library `audiomentations`
* details in https://hackmd.io/qd203WA3T9uq6TtmP2yTdw?both#New-model-implementation

Now, implement two kinds:  
1. Add white noise `AddGaussianNoise`:   
run `create_audio_dataset_add_noise.py`  
three kinds of SNR(db)
    * 15 db
    * 20 db
    * 35 db
2. `TimeStretch` : scale 0.8~1.25   
run `create_audio_dataset_timestretch.py`   

Future will implement below methods:  
3. TimeMasking  
4. TimeShifting

