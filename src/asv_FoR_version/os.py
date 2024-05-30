import os
source_folder = r"D:\FoR\FoR_for_norm\for-norm\training\real"
file = open("A.txt", "w")

file_list = os.listdir(source_folder)
for i in file_list:
    file.write(i)

file.close()

file = open(r"D:\graduate_project\src\asv_FoR_version\training_detail_model5_add_FoR.txt", "w")
file.close()