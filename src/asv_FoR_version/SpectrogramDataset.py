from PIL import Image
import numpy as np
import torch

class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames    # 資料集的所有檔名
        self.labels = labels          # 影像的標籤
        self.transform = transform    # 影像的轉換方式
 
    def __len__(self):
        return len(self.filenames)    # return DataSet 長度
 
    def __getitem__(self, idx):       # idx: Inedx of filenames
        # Transform image
        image = self.transform(Image.fromarray((self._remove_margin(self.filenames[idx]))[1].astype('uint8')).convert('RGB'))
        label = np.array(self.labels[idx])
        return image, label           # return 模型訓練所需的資訊
    
    def _remove_margin(self, image_path):
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