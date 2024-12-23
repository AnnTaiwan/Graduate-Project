#!/bin/sh
# used for CH dataset
# define variables
INPUT_FILE=$1
OUTPUT_FILE=$2
NOISE_PROFILE=$3  # 一個包含背景噪音的檔案，從中獲取噪音特徵
SAMPLE_RATE=22050 # fit how sample on mel-spectrogram code

# 使用 SoX 生成噪音特徵 
# fetch 0~1 sec from input file and use noiseprof to generate noised audio.
sox "$INPUT_FILE" -n trim 0 0.5 noiseprof "$NOISE_PROFILE"

# 使用 SoX 進行降噪
# noisered "$NOISE_PROFILE" 0.25 : we use noised audio to do denoise and "0.25" is the parameter.
# rate "$SAMPLE_RATE" : specify output file 's sample rate to $SAMPLE_RATE.
# --norm=-1 : help us do audio volume enhancement.
#sox --norm=-1 "$INPUT_FILE" "$OUTPUT_FILE" rate "$SAMPLE_RATE" noisered "$NOISE_PROFILE" 0.000000005

sox  --norm=-1 "$INPUT_FILE" "$OUTPUT_FILE" noisered "$NOISE_PROFILE" 0.005
#sox  "$INPUT_FILE" "$OUTPUT_FILE" noisered "$NOISE_PROFILE" 0.00130