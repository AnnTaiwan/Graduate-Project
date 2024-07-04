#!/bin/bash

echo "####################################"
echo "## This is used for CHINESE model ##"
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

echo "## (2) Prepare to convert the audio into mel-spectrogram..."
python3 create_spectrogram.py
echo "Finish converting the audio, and the mel-spectrogram is prepared."

echo "## (3) Predicting the input audio..."
./demo_final_segment_audio /usr/share/vitis_ai_library/models/CNN_model8_netname/CNN_model8_netname.xmodel spec/ out_spec_sh.txt
echo
echo "Finish predicting the audio, and now you can see the result on the LINE."




