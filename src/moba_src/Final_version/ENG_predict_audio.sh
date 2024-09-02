#!/bin/bash

echo "####################################"
echo "## This is used for ENGLISH model ##"
echo "####################################"

# Define a variable to indicate if recording is done
recording_done=false

# Function to capture Ctrl+C signal and stop recording
stop_recording() {
    echo "## STOP RECORDING ##"
    kill $PID
    recording_done=true
}

# Capture Ctrl+C signal (SIGINT)
trap stop_recording SIGINT

# Start recording
echo "## (1) Now you can rercord an audio, intput ctrl+c to end recording..."
gst-launch-1.0 alsasrc device=hw:0 ! audioconvert ! wavenc ! filesink location=audio/audio1.wav &
PID=$!

# Wait for the recording process to complete
while [ "$recording_done" = false ]; do
    sleep 1
done

# Ensure the recording process has stopped
wait $PID

# Prepare folders for input and output
AUDIO_FOLDER="./audio"             # The folder where audio files are stored
TXT_FOLDER="./mel_spec_txt"         # The folder where mel spectrogram txt files will be saved
IMG_FOLDER="./mel_spec_images"      # The folder where mel spectrogram images will be saved

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
echo "## Finish converting all audio files, and mel-spectrogram images are prepared."

echo "## (3) Predicting the input audio..."
./demo_final_segment_audio CNN_model13_ENG_netname.xmodel $IMG_FOLDER/ detail_result_ENG.txt
echo
echo "Finish predicting the audio, and now you can see the result on the LINE."
echo
echo "## Statistics"
cat Result.txt
echo
# ./client_ver2
