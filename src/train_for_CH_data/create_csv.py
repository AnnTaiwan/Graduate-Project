'''
update csv info for chinese data
'''
import pandas as pd
import os
dict_format = {
    "file_name":[],
    "label":[]
}

bonafide_dir = r"D:\clone_audio\chinese_audio_spec\bonafide_spec_total"
spoof_dir = r"D:\clone_audio\chinese_audio_spec\spoof_spec_total"
spoof_dir2 = r"D:\clone_audio\chinese_audio_spec\test_spoof_spec_suno_bark202_241"
for i in os.listdir(bonafide_dir):
    dict_format["file_name"].append(i)
    dict_format["label"].append(0)
for i in os.listdir(spoof_dir):
    dict_format["file_name"].append(i)
    dict_format["label"].append(1)
for i in os.listdir(spoof_dir2):
    dict_format["file_name"].append(i)
    dict_format["label"].append(1)
df = pd.DataFrame(dict_format)
df.to_csv('chinese_audio_info.csv', index=False)