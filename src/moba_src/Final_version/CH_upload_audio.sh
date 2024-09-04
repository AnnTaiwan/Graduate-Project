#!/bin/bash

echo "################################################"
echo "## This is used for predicting uploaded audio ##"
echo "################################################"


# Prepare folders for input and output
AUDIO_FOLDER="./upload_audio"                               # The folder where original audio files are stored
NOISE_REDUCE_AUDIO_FOLDER="./noise_reduce_audio"     # The folder where NOISE-REDUCE audio files are stored
FIX_AUDIO_FOLDER="./fix_audio"                       # The folder where fix audios are saved   
TXT_FOLDER="./mel_spec_txt"                          # The folder where mel spectrogram txt files will be saved
IMG_FOLDER="./mel_spec_images_denosie"                       # The folder where mel spectrogram images will be saved
PROFILE="./profile"                                  # The folder where the profile.wav needed by denoise-step and recording the noise is saved


# Create output folders if they don't exist
mkdir -p $NOISE_REDUCE_AUDIO_FOLDER
mkdir -p $FIX_AUDIO_FOLDER
mkdir -p $TXT_FOLDER
mkdir -p $IMG_FOLDER
mkdir -p $PROFILE

# let the folder be empty
rm $TXT_FOLDER/*
rm $IMG_FOLDER/*
rm $NOISE_REDUCE_AUDIO_FOLDER/*
rm $FIX_AUDIO_FOLDER/*
rm $PROFILE/*


echo "## (2) Fix the audio, and saving the revised audio into $FIX_AUDIO_FOLDER/ "
sox $AUDIO_FOLDER/audio1.wav $FIX_AUDIO_FOLDER/audio_fix.wav

echo "## (3) Reduce the nosie and increase volume in audio_fix.wav, and saving the denoised-audio into $NOISE_REDUCE_AUDIO_FOLDER/ "
sh denoise_add_volume.sh $FIX_AUDIO_FOLDER/audio_fix.wav $NOISE_REDUCE_AUDIO_FOLDER/audio_denoise.wav $PROFILE/audio_profile.prof
#sh denoise_add_volume.sh $AUDIO_FOLDER/audio1.wav $NOISE_REDUCE_AUDIO_FOLDER/audio_denoise.wav $PROFILE/audio_profile.wav


echo "## (4) Convert audio files to mel-spectrogram text files..."
# Iterate over each file in the audio folder
for audio_file in "$NOISE_REDUCE_AUDIO_FOLDER"/*; do
    # Run the mel-spectrogram calculation for each file
    ./cal_mel_spec_ver4 "$audio_file" "$TXT_FOLDER"
    echo "Generated txt files in $TXT_FOLDER from $audio_file"
done

echo "## (5) Convert mel-spectrogram text files to images..."

# Iterate over each .txt file in the txt folder
for txt_file in "$TXT_FOLDER"/*.txt; do
    # Get the base name of the txt file (without the directory and extension)
    base_name=$(basename "$txt_file" .txt)
    # Correct string concatenation for the image name
    image_name="image_$base_name"
    
    # Run the plot script for each txt file and save the image
    #./plot_mel_spec_from_txt_scale_ver2 "$txt_file" "$IMG_FOLDER/$image_name.png"
    ./plot_mel_spec_from_txt_ver2 "$txt_file" "$IMG_FOLDER/$image_name.png"
    echo "Generated: $IMG_FOLDER/$image_name.png"
done

echo "## Finish converting all audio files, and mel-spectrogram images are prepared."

echo "## (6) Predicting the input audio..."
./demo_final_segment_audio_ver2 CNN_model9_CH_ver3_netname.xmodel $IMG_FOLDER/ detail_result_CH.txt
echo
echo "Finish predicting the audio, and now you can see the result on the LINE."
echo
echo "## Statistics"
cat Result.txt
echo
# ./client_ver2



