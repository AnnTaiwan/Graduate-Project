'''
By using this code, I can view the original image (displayed using plt.imshow) 
and use the mouse to find the suitable coordinates to crop the image, excluding the margin.
'''
from P_model_9_CH import crop_with_points
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

filepath = r"D:\clone_audio\chinese_audio_dataset_ver3\test_mel_spec\bonafide_melspec_SSB01330093.png"
img = Image.open(filepath)
cropped_image = crop_with_points(filepath).convert('RGB')

# Display the image before cropped
print("Before Cropped:")
plt.imshow(img, cmap='gray')
# plt.axis('off')
plt.show()
print(np.array(img).shape)

# Display the cropped image
print("After Cropped:")
plt.imshow(cropped_image, cmap='gray')
# plt.axis('off')
plt.show()

print(np.array(cropped_image).shape)