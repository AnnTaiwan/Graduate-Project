from flask import Flask, render_template, jsonify, request
import threading
import subprocess
import time
import os

app = Flask(__name__)

# Global flags for control
stop_recording_event = threading.Event()
stop_processing_event = threading.Event()

# Variables to store current audio files and results
audio_files = []
audio_history = []  # For storing history of audio files
result_text = ""
result_history = []  # For storing history of result text
dealing_audio_text = ""

# Process references for later termination
recording_process = None
processing_process = None
# predicted language mode
mode = "CH"
# Function to update audio files in the folder
def update_audio_files():
    global audio_files
    while not stop_recording_event.is_set():  # Check only while recording
        new_files = [f for f in os.listdir("../audio") if os.path.isfile(os.path.join("../audio", f))]
        
        # Append to history if there's a change
        if new_files != audio_files:
            audio_files = new_files
            audio_history.append(list(audio_files))
        
        time.sleep(1)  # Check every second

# Function to monitor current_dealing_audio.txt for changes
def monitor_dealing_audio():
    global dealing_audio_text
    last_content = ""
    
    while not stop_processing_event.is_set():  # Check only while processing
        if os.path.exists("../current_dealing_audio.txt"):
            with open("../current_dealing_audio.txt", "r") as file:
                current_content = file.read()
                
                if current_content != last_content:  # Check if content has changed
                    last_content = current_content
                    dealing_audio_text = current_content  # Update global variable
        time.sleep(1)  # Check every second

# Function to monitor Result.txt for real-time updates
def monitor_result_text():
    global result_text, result_history, dealing_audio_text
    last_combined_result = ""
    while not stop_processing_event.is_set():  # Check only while processing
        if os.path.exists("../Result.txt"):
            with open("../Result.txt", "r") as file:
                current_content = file.read()

                # Combine current dealing audio with the result
                combined_text = f"Audio: {dealing_audio_text} - Result: {current_content}"
                
                if last_combined_result != combined_text:
                    last_combined_result = combined_text
                    result_text = combined_text
                    result_history.append(result_text)  # Append to result history
        time.sleep(1)  # Check every second

# Function to record audio
def record_audio():
    global recording_process
    stop_recording_event.clear()  # Ensure recording starts from scratch
    update_audio_files_thread = threading.Thread(target=update_audio_files)
    update_audio_files_thread.start()  # Start the thread to update audio files

    file_index = 1  # Initialize the file index for naming audio files

    while not stop_recording_event.is_set():
        # Command to record audio for 5 seconds
        command = f"gst-launch-1.0 alsasrc device=hw:0 ! audioconvert ! wavenc ! filesink location=\"../audio/audio_{file_index}.wav\""
        
        # Start the recording process
        recording_process = subprocess.Popen(command, shell=True)
        
        time.sleep(6)  # Record for 6 seconds
        
        # Check if recording_process is not None before terminating
        if recording_process is not None:
            recording_process.terminate()  # Terminate the recording process
            
        file_index += 1  # Increment the file index for the next recording

    # Clean up the thread when done
    update_audio_files_thread.join()

# Function to process audio
def process_audio_CH():
    global processing_process
    stop_processing_event.clear()  # Ensure processing starts from scratch

    # Start monitoring dealing audio and result text in separate threads
    threading.Thread(target=monitor_dealing_audio).start()
    threading.Thread(target=monitor_result_text).start()

    # Execute the processing shell script
    processing_process = subprocess.Popen("sh ../CH_predict_audio_real_time.sh", shell=True)
    processing_process.wait()  # Wait for the process to finish
    stop_processing_event.set()  # Set the stop flag to true when processing finishes

# Function to process audio
def process_audio_ENG():
    global processing_process
    stop_processing_event.clear()  # Ensure processing starts from scratch

    # Start monitoring dealing audio and result text in separate threads
    threading.Thread(target=monitor_dealing_audio).start()
    threading.Thread(target=monitor_result_text).start()

    # Execute the processing shell script
    processing_process = subprocess.Popen("sh ../ENG_predict_audio_real_time.sh", shell=True)
    processing_process.wait()  # Wait for the process to finish
    stop_processing_event.set()  # Set the stop flag to true when processing finishes


# Start recording
@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording_process
    if recording_process is None:  # Allow starting if not running
        stop_recording_event.clear()  # Clear the stop event
        recording_thread = threading.Thread(target=record_audio)
        recording_thread.start()
        return jsonify(status="Recording started", is_recording=True)
    else:
        return jsonify(status="Recording already in progress", is_recording=False)

# Start processing
@app.route('/start_processing', methods=['POST'])
def start_processing():
    global processing_process
    if processing_process is None:  # Allow starting if not running
        stop_processing_event.clear()  # Clear the stop event
        if(mode == "CH"):
            processing_thread = threading.Thread(target=process_audio_CH)
            processing_thread.start()
        else:
            processing_thread = threading.Thread(target=process_audio_ENG)
            processing_thread.start()
        return jsonify(status="Processing started", is_processing=True)
    else:
        return jsonify(status="Processing already in progress", is_processing=False)

# Stop recording
@app.route('/stop_recording', methods=['POST'])
def stop_recording_endpoint():
    global recording_process
    stop_recording_event.set()
    
    if recording_process is not None:  # Check if recording_process is not None before terminating
        recording_process.terminate()  # Kill the recording process
        recording_process = None  # Clear the reference

    return jsonify(status="Recording stopped")

# Stop processing
@app.route('/stop_processing', methods=['POST'])
def stop_processing_endpoint():
    global processing_process
    stop_processing_event.set()
    
    if processing_process:
        processing_process.terminate()  # Kill the processing process
        processing_process = None  # Clear the reference
        
    return jsonify(status="Processing stopped")

# Clear history
@app.route('/clear_history', methods=['POST'])
def clear_history():
    global audio_files, audio_history, result_text, result_history, dealing_audio_text
    global recording_process, processing_process

    # Clear variables
    audio_files = []
    audio_history = []
    result_text = ""
    result_history = []
    dealing_audio_text = ""

    # Stop and reset processes
    stop_recording_event.set()
    stop_processing_event.set()

    if recording_process is not None:
        recording_process.terminate()
        recording_process = None

    if processing_process is not None:
        processing_process.terminate()
        processing_process = None

    # Reset event flags to allow new operations
    stop_recording_event.clear()  # Allow new recording
    stop_processing_event.clear()  # Allow new processing

    return jsonify(status="History cleared and reset")

# Get current audio files and history
@app.route('/audio_files', methods=['GET'])
def get_audio_files():
    return jsonify(files=audio_files, history=audio_history)

# Get the result text and history
@app.route('/result', methods=['GET'])
def get_result():
    return jsonify(result=result_text, history=result_history, dealing_audio=dealing_audio_text)

# Exit current mode safely
@app.route('/exit_mode', methods=['POST'])
def exit_mode():
    global audio_files, audio_history, result_text, result_history, dealing_audio_text
    global recording_process, processing_process

    # Clear variables
    audio_files = []
    audio_history = []
    result_text = ""
    result_history = []
    dealing_audio_text = ""

    stop_recording_event.set()
    stop_processing_event.set()

    if recording_process is not None:
        recording_process.terminate()
        recording_process = None

    if processing_process is not None:
        processing_process.terminate()
        processing_process = None

    # Reset event flags to allow new operations
    stop_recording_event.clear()  # Allow new recording
    stop_processing_event.clear()  # Allow new processing
    
    return jsonify(status="Exited mode and stopped all processes")

# Home page
@app.route('/')
def index():
    return render_template('index.html')
# CH page
@app.route('/CH_predict')
def index_CH():
    global mode
    mode = "CH"
    return render_template('index_CH_predict.html')
# ENG page
@app.route('/ENG_predict')
def index_ENG():
    global mode
    mode = "ENG"
    return render_template('index_ENG_predict.html')

if __name__ == '__main__':
    app.run(host='192.168.137.159', port=5000, debug=True)
