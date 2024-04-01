from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import os

from P_model_5_asv_FoR import IMAGE_SIZE, CNN_model5
from SpectrogramDataset import SpectrogramDataset

BATCH_SIZE_TEST = 500

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

# Transformer
test_transformer = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    # transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # normalize
])
 

def get_test_dataloader(fake_image_folder_path, real_image_folder_path):
    # Read fake one
    image_paths = [os.path.join(fake_image_folder_path, filename) for filename in os.listdir(fake_image_folder_path)[2500:5000]]
    # Load Labels
    labels = [1.0 for _ in os.listdir(fake_image_folder_path)[2500:5000]]

    # Read real one
    image_paths = image_paths + [os.path.join(real_image_folder_path, filename) for filename in os.listdir(real_image_folder_path)[2500:5000]]
    # Load Labels
    labels = labels + [0.0 for _ in os.listdir(real_image_folder_path)[2500:5000]]
    
    # Dataset parameters: (self, all filenames, all labels, transform)
    test_dataloader = torch.utils.data.DataLoader(SpectrogramDataset(image_paths, labels, test_transformer),
                                  batch_size = BATCH_SIZE_TEST, shuffle = False, pin_memory=True)
    return test_dataloader

def test_model(model, test_dataloader, criterion):
    model.eval()  # 設定為評估模式
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 不需要計算梯度
        for _, (image, label) in enumerate(test_dataloader):
            image, label = image.to(device), label.to(device)
            output = model(image)
            loss = criterion(output, label.long())
            test_loss += loss.item() * image.size(0)

            probs = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(probs, 1)

            total += label.size(0)
            correct += (predicted == label).sum().item()

    avg_loss = test_loss / total
    accuracy = 100 * correct / total

    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    # 決定要在CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Test on {device}.")
    # Load the model and weights
    model = CNN_model5()
    # Move model to device
    model.to(device)
    # state_dict = torch.load("model_5.h5")
    state_dict = torch.load(r"D:\graduate_project\src\asv_FoR_version\model_5_asv_FoR_mix.h5")
    model.load_state_dict(state_dict)
    # set loss function
    criterion = nn.CrossEntropyLoss()

    # Load Images from a Folder
    fake_image_folder_path = r"D:\FoR\FoR_for_norm\for-norm\training\spec\spec_fake"
    real_image_folder_path = r"D:\FoR\FoR_for_norm\for-norm\training\spec\spec_real"
    # Load images into test_dataloader
    test_dataloader = get_test_dataloader(fake_image_folder_path, real_image_folder_path)
    print(f"Loaded test data from {fake_image_folder_path}\n and {real_image_folder_path}.")

    # Start testing
    print("Testing...")
    test_model(model, test_dataloader, criterion)