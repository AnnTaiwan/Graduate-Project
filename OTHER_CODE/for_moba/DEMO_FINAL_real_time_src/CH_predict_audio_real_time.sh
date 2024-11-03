#!/bin/bash

cd ../
echo "####################################"
echo "## This is used for CHINESE model ##"
echo "####################################"

# Prepare folders for input and output
AUDIO_FOLDER="./audio"                               # The folder where original audio files are stored

NOISE_REDUCE_AUDIO_FOLDER="./noise_reduce_audio"     # The folder where NOISE-REDUCE audio files are stored
FIX_AUDIO_FOLDER="./fix_audio"                       # The folder where fix audios are saved   
TXT_FOLDER="./mel_spec_txt"                          # The folder where mel spectrogram txt files will be saved
IMG_FOLDER="./mel_spec_images_denosie"               # The folder where mel spectrogram images will be saved
PROFILE="./profile"                                  # The folder where the profile.wav needed by denoise-step and recording the noise is saved

PROCESSED_AUDIO="./processed_audio"                  # The folder where contains audio that is already predicted

# Create output folders if they don't exist
mkdir -p $NOISE_REDUCE_AUDIO_FOLDER
mkdir -p $FIX_AUDIO_FOLDER
mkdir -p $TXT_FOLDER
mkdir -p $IMG_FOLDER
mkdir -p $PROFILE
mkdir -p $PROCESSED_AUDIO

rm -rf $PROCESSED_AUDIO/*
rm current_dealing_audio.txt
# Infinite loop to process files until stopped
while true; do
    # Check if there are any audio files in the folder
    if ls $AUDIO_FOLDER/*.wav 1> /dev/null 2>&1; then
        for audio_file in $AUDIO_FOLDER/*.wav; do
            echo $audio_file > current_dealing_audio.txt
            # let the output folders be empty
            rm -rf $TXT_FOLDER/*
            rm -rf $IMG_FOLDER/*
            rm -rf $NOISE_REDUCE_AUDIO_FOLDER/*
            rm -rf $FIX_AUDIO_FOLDER/*
            rm -rf $PROFILE/*
            
            # Processing the detected audio file
            echo "## (2) Fixing the audio $audio_file, saving the revised audio into $FIX_AUDIO_FOLDER/ "
            # sox -t wav $audio_file -r 22050 -b 16 $FIX_AUDIO_FOLDER/audio_fix.wav channels 1
            sox  $audio_file $FIX_AUDIO_FOLDER/audio_fix.wav
            echo "## (3) Reducing noise and increasing volume in audio_fix.wav, saving into $NOISE_REDUCE_AUDIO_FOLDER/ "
            sh denoise_add_volume_CH.sh $FIX_AUDIO_FOLDER/audio_fix.wav $NOISE_REDUCE_AUDIO_FOLDER/audio_denoise.wav $PROFILE/audio_profile.prof

            echo "## (4) Converting audio files to mel-spectrogram text files..."
            for denoised_file in "$NOISE_REDUCE_AUDIO_FOLDER"/*; do
                ./cal_mel_spec_ver4 $denoised_file $TXT_FOLDER
                echo "Generated txt files in $TXT_FOLDER from $denoised_file"
            done

            echo "## (5) Converting mel-spectrogram text files to images..."
            for txt_file in "$TXT_FOLDER"/*.txt; do
                base_name=$(basename "$txt_file" .txt)
                image_name="image_$base_name"
                ./plot_mel_spec_from_txt_ver2 "$txt_file" "$IMG_FOLDER/$image_name.png"
                echo "Generated: $IMG_FOLDER/$image_name.png"
            done

            echo "## Finish converting $audio_file, mel-spectrogram images prepared."

            echo "## (6) Predicting the input audio..."
            ./demo_final_segment_audio_ver2 CNN_model9_CH_ver4_netname.xmodel $IMG_FOLDER/ detail_result_CH.txt
            echo "Finished predicting $audio_file."
            echo "## Statistics"
            cat Result.txt
            echo
            cat detail_result_CH.txt

            # Extract the base name of the audio file (without directory)
            audio_name=$(basename "$audio_file")
            
            # Construct the new file path in the processed folder
            new_audio_name="$PROCESSED_AUDIO/$audio_name"
            
            # After processing, move the processed file to the new location
            mv "$audio_file" "$new_audio_name"
            # ./client_ver2
            sleep 2  
        done
    else
        echo "No audio files found. Waiting..."
        sleep 5  # Sleep for 5 seconds before checking again
    fi
done
