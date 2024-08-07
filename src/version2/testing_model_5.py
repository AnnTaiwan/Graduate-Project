from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import os

from P_model_5 import IMAGE_SIZE, CNN_model5_small
from SpectrogramDataset import SpectrogramDataset

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

BATCH_SIZE_TEST = 50

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

# Transformer
test_transformer = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    # transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # normalize
])
 

def get_test_dataloader(data_dir):
    # save the file path
    test_image_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
    # Load Labels
    # Assuming 'eval_df' has columns 'filename' and 'target'
    # skip the "spec_" and ".png"
    test_labels = [dev_df[dev_df["filename"] == os.path.basename(path)[5:-4]]["target"].values[0] for path in test_image_paths]
    
    # Dataset parameters: (self, all filenames, all labels, transform)
    test_dataloader = torch.utils.data.DataLoader(SpectrogramDataset(test_image_paths, test_labels, test_transformer),
                                  batch_size = BATCH_SIZE_TEST, shuffle = False, pin_memory=True)
    return test_dataloader

def test_model(model, test_dataloader, criterion):
    model.eval()  # 設定為評估模式
    test_loss = 0.0
    correct = 0
    total = 0
    y_true = np.array([])
    y_pred = np.array([])
    with torch.no_grad():  # 不需要計算梯度
        for _, (image, label) in enumerate(test_dataloader):
            image, label = image.to(device), label.to(device)
            output = model(image)
            loss = criterion(output, label)
            test_loss += loss.item() * image.size(0)

            probs = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(probs, 1)

            # 將 CUDA 張量轉換為 NumPy 陣列
            y_pred = np.append(y_pred, predicted.cpu().numpy())
            y_true = np.append(y_true, label.cpu().numpy())

            total += label.size(0)
            # correct += (predicted == label).sum().item()

    avg_loss = test_loss / total
    # accuracy = 100 * correct / total

    print(f"Test Loss: {avg_loss:.4f}")

    # Calculate precision
    precision = precision_score(y_true, y_pred)

    # Calculate recall
    recall = recall_score(y_true, y_pred)

    # Calculate F1-score
    f1 = f1_score(y_true, y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    print(f"Precision: {precision * 100:.2f}")
    print(f"Recall: {recall * 100:.2f}")
    print(f"F1-score: {f1 * 100:.2f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    # 決定要在CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Test on {device}.")
    # 讀取LA_dev_info
    dev_df = pd.read_csv("eval_info.csv")
    # Load the model and weights
    model = CNN_model5_small()
    # Move model to device
    model.to(device)
    # state_dict = torch.load("model_5.h5")
    state_dict = torch.load(r"D:\graduate_project\src\version2\model_5_Large.h5")
    model.load_state_dict(state_dict)
    # set loss function
    criterion = nn.CrossEntropyLoss()

    # Load Images from a Folder
    image_folder_path = r"D:/graduate_project/src/spec_LAEval_audio_shuffle3_NOT_preprocessing"
    # Load images into test_dataloader
    test_dataloader = get_test_dataloader(image_folder_path)
    print(f"Loaded test data from {image_folder_path}.")

    # Start testing
    print("Testing...")
    test_model(model, test_dataloader, criterion)