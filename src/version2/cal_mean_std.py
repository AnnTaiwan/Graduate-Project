'''
Used for calculating the mean and std of train data, but not used in the end.
'''
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import os

def compute_mean_std(image_folder_path):
    image_paths = [os.path.join(image_folder_path, filename) for filename in os.listdir(image_folder_path)]

    # Define a transform to resize and convert images to tensors
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to 128x128
        transforms.ToTensor()
    ])

    mean = np.zeros(3)
    std = np.zeros(3)
    num_pixels = 0

    for path in image_paths:
        # Load image using PIL
        image_pil = Image.open(path).convert('RGB')  # Ensure it's RGB
        image_tensor = transform(image_pil)
        
        # Accumulate sum and sum of squares of pixel values
        mean += image_tensor.mean(axis=(1, 2)).numpy()
        std += image_tensor.std(axis=(1, 2)).numpy()
        num_pixels += image_tensor.shape[1] * image_tensor.shape[2]

    mean /= len(image_paths)
    std /= len(image_paths)
    std = np.sqrt(std - mean ** 2)

    return mean, std

if __name__ == "__main__":
    # Specify the path to your image folder
    image_folder_path = r"D:\graduate_project\src\spec_LATrain_audio_shuffle23_NOT_preprocessing"

    # Calculate mean and std
    mean, std = compute_mean_std(image_folder_path)

    print("Mean:", mean)
    print("Std:", std)
