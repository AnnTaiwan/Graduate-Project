import torch
import torch.nn as nn
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import os

from P_model_13_CH import IMAGE_SIZE, CNN_model13
from SpectrogramDataset import SpectrogramDataset

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

BATCH_SIZE_TEST = 50


dict1 = {
	"Image_name":[],
	"output_0":[],
	"output_1":[]
}

# Transformer
test_transformer = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])
 

def get_test_dataloader(data_dir):
    # save the file path
    test_image_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]

    # Load Labels
    # spoof is 1, bonafide is 0, see teh first character
    test_labels = [1 if filename[0] == "s" else 0 for filename in os.listdir(data_dir)]


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
            image, label = image.to(device).float(), label.to(device).long()
            output = model(image)
            loss = criterion(output, label)
            test_loss += loss.item() * image.size(0)

			# record the outputs of each columns
            dict1["output_0"].extend(output[:, 0].tolist())
            dict1["output_1"].extend(output[:, 1].tolist())
            
            # print(output)
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
    # Load the model and weights
    model = CNN_model13()
    # Move model to device
    model.to(device)
    pth_path = 'training_result_model13_CH/model_13_CH_ver1.pth'
    state_dict = torch.load(pth_path)
    model.load_state_dict(state_dict)
    print(f"Load pth from {pth_path}")
    # set loss function
    criterion = nn.CrossEntropyLoss()

    # Load Images from a Folder
    image_folder_path = r"D:\clone_audio\chinese_audio_dataset_ver3\test_mel_spec_padding_original_audio"
    
    # record the Image_name
    dict1["Image_name"] = [filename for filename in os.listdir(image_folder_path)]
    # Load images into test_dataloader
    test_dataloader = get_test_dataloader(image_folder_path)
    print(f"Loaded test data from {image_folder_path}.")

    # Start testing
    print("Testing...")
    test_model(model, test_dataloader, criterion)
    
    # turn dict1 into csv file
    output_df = pd.DataFrame(dict1)
    csv_file_path = f"training_result_model13_CH/testing_result13_CH_ver1.csv"
    output_df.to_csv(csv_file_path, index=False)
    print(f"Data has been successfully exported to {csv_file_path}")