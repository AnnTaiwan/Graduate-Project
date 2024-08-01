from PIL import Image
import numpy as np
import torch

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, mel_spec_path, mfcc_path, chroma_path, labels, transform):
        self.mel_spec_path = mel_spec_path   # Mel_spectrogram total paths 
        self.mfcc_path = mfcc_path           # mfcc total paths
        self.chroma_path = chroma_path       # chroma total paths
        self.labels = labels                 # total labels
        self.transform = transform           # image transform

    def __len__(self):
        return len(self.mel_spec_path)    # return DataSet size

    def __getitem__(self, idx):       # idx: Index of filenames
        # Transform image
        mel_spec_image = self.transform(self._crop_with_points(self.mel_spec_path[idx]).convert('RGB')) 
        mfcc_image = self.transform(self._crop_with_points(self.mfcc_path[idx]).convert('RGB')) 
        chroma_image = self.transform(self._crop_with_points(self.chroma_path[idx]).convert('RGB')) 
        # Stack images along the channel dimension
        stacked_image = torch.cat([mel_spec_image, mfcc_image, chroma_image],  dim=0) 
        label = np.array(self.labels[idx])
        return stacked_image, label           

    def _crop_with_points(self, image_path):
        points = [(79, 57), (575, 428), (575, 57), (79, 428)]
        # Load the image
        img = Image.open(image_path)
        # Define the four points
        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]
        x4, y4 = points[3]

        # Find the bounding box for cropping
        left = min(x1, x4)
        upper = min(y1, y2)
        right = max(x2, x3)
        lower = max(y3, y4)
        # Crop the image
        cropped_img = img.crop((left, upper, right, lower))

        return cropped_img

