#!/bin/bash

echo "####################################"
echo "## This is used for DRAW MEL_SPEC ##"
echo "####################################"


# Prepare folders for input and output
AUDIO_FOLDER=$1                           # The folder where audio files are stored
TXT_FOLDER="./mel_spec_txt"               # The folder where mel spectrogram txt files will be saved
IMG_FOLDER=$2            # The folder where mel spectrogram images will be saved

# Create output folders if they don't exist
mkdir -p $TXT_FOLDER
mkdir -p $IMG_FOLDER

# let the folder be empty
rm $TXT_FOLDER/*
rm $IMG_FOLDER/*



echo "## (1) Convert audio files to mel-spectrogram text files..."

# Iterate over each file in the audio folder
for audio_file in "$AUDIO_FOLDER"/*; do
    # Run the mel-spectrogram calculation for each file
    ./cal_mel_spec_ver4 "$audio_file" "$TXT_FOLDER"
    echo "Generated txt files in $TXT_FOLDER from $audio_file"
done

echo "## (2) Convert mel-spectrogram text files to images..."

# Iterate over each .txt file in the txt folder
for txt_file in "$TXT_FOLDER"/*.txt; do
    # Get the base name of the txt file (without the directory and extension)
    base_name=$(basename "$txt_file" .txt)
    # Correct string concatenation for the image name
    image_name="image_$base_name"
    
    # Run the plot script for each txt file and save the image
    ./plot_mel_spec_from_txt_ver2 "$txt_file" "$IMG_FOLDER/$image_name.png"
    echo "Generated: $IMG_FOLDER/$image_name.png"
done