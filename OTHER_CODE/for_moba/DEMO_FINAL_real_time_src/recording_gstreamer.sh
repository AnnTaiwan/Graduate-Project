#!/bin/bash

###################################### RECORDING ###############################################################
# Prepare folders for input and output
AUDIO_FOLDER="./audio"  # The folder where original audio files are stored

# Create audio folder if it doesn't exist
mkdir -p $AUDIO_FOLDER

# Clean existing audio files
rm -rf $AUDIO_FOLDER/*

# Initialize file index and set recording segment duration (in seconds)
SEGMENT_DURATION=6
file_index=1  # Start file index from 1

# Define a variable to indicate if recording is done
recording_done=false
last_pid=0  # Keep track of the last recording process PID

# Function to capture Ctrl+C signal and stop recording safely
stop_recording() {
    echo "## STOP RECORDING ##"
    recording_done=true
    
    # Wait for the last recording to finish and ensure it is properly saved
    if [ $last_pid -ne 0 ]; then
        echo "## Finishing the last recording segment ##"
        kill $last_pid
    fi
}

# Capture Ctrl+C signal (SIGINT)
trap stop_recording SIGINT

# Loop to continuously record 5-second segments
while [ "$recording_done" = false ]; do
    echo "## (1) Recording segment $file_index ..."

    # Record audio for SEGMENT_DURATION seconds and save to a file
    gst-launch-1.0 alsasrc device=hw:0 ! \
        audioconvert ! \
        wavenc ! \
        filesink location="$AUDIO_FOLDER/audio_$file_index.wav" &

    last_pid=$!

    # Wait for SEGMENT_DURATION seconds
    sleep $SEGMENT_DURATION

    # Ensure the recording process has stopped before starting the next one
    kill $last_pid

    # Increment the file index for the next segment
    file_index=$((file_index + 1))
done

echo "## Recording complete. Files saved in $AUDIO_FOLDER ##"

###################################### RECORDING END ###########################################################
