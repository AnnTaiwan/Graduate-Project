'''
By using this code, I can view the original image (displayed using plt.imshow) 
and use the mouse to find the suitable coordinates to crop the image, excluding the margin.
'''
from P_model_9_CH import crop_with_points
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

filepath = r"D:\graduate_project\OTHER_CODE\for_moba\August_21_testing\mel_spectrogram.png"
# filepath = r"D:\graduate_project\feature_extraction_src\three.png"

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