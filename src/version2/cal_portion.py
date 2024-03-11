import pandas as pd
import os
df = pd.read_csv("eval_info.csv")
path = r"D:\graduate_project\src\spec_LAEval_audio_shuffle3_NOT_preprocessing"
list_image_name = [filename[5:-4] for filename in os.listdir(path)]
spoof = 0
bonafide = 0
for i in list_image_name:
    if any(df.loc[df["filename"]==i, "target"]):
        spoof += 1
    else:
        bonafide += 1

print(f"Spoof: {spoof}, Bonafide: {bonafide}")
