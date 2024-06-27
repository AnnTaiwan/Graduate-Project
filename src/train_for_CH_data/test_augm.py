'''
Test on data augmentation by transformming
'''
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
# 加载图像
path = r"D:\clone_audio\chinese_audio_spec\spoof_spec\spoof_ElevenLabs_2024-03-31T18_30_02_Thomas_pre_s50_sb75_se0_b_m2.png"
image = Image.open(r"D:\clone_audio\chinese_audio_spec\spoof_spec\spoof_ElevenLabs_2024-03-31T18_30_02_Thomas_pre_s50_sb75_se0_b_m2.png")
plt.imshow(image)
plt.axis('off')  # 关闭坐标轴
plt.show()
# 定义增强变换
transform = transforms.Compose([
        transforms.Resize((256, 256)), # Original Size: (640, 480)
        transforms.ToTensor()
    ])
# do data augmentation on spoof data
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomAffine(degrees=0, translate=(0.2, 0), scale=None, shear=0),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
# 移除圖片周圍空白處
def remove_margin(image_path):
    # Load the image
    img = Image.open(image_path)
    img2 = np.array(img)

    # Find the bounding box of the non-background region
    non_bg_indices = np.argwhere(img2 != 255)  # Assuming the background is white (255)
    min_row, min_col, _ = np.min(non_bg_indices, axis=0)
    max_row, max_col, _ = np.max(non_bg_indices, axis=0)

    # Crop the image to the bounding box
    cropped_img = img2[min_row + 25:max_row-38, min_col + 61:max_col+1] # 自行調數值刪掉空白

    return img, cropped_img
# 应用增强
image_aug = augment_transform(Image.fromarray(remove_margin(path)[1].astype('uint8')).convert('RGB'))

# 转换回PIL图像
image_aug_pil = transforms.ToPILImage()(image_aug)

# 显示图像
plt.imshow(image_aug_pil)
plt.axis('off')  # 关闭坐标轴
plt.show()

# image_aug_pil = transforms.ToPILImage()(image)
# # 显示图像
# plt.imshow(image_aug_pil)
# plt.axis('off')  # 关闭坐标轴
# plt.show()