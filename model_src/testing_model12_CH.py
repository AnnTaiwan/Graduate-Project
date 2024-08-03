import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

from P_model_12_CH import CNN_model12
from AudioDataset_ver2 import AudioDataset_ver2

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

BATCH_SIZE_TEST = 50


dict1 = {
	"Audio_name":[],
	"output_0":[],
	"output_1":[]
}

def get_test_dataloader(data_dir_bonafide, data_dir_spoof):
    # save the bonafide file path
    test_audio_paths = [os.path.join(data_dir_bonafide, filename) for filename in os.listdir(data_dir_bonafide)]
    test_labels = [0 for _ in range(len(os.listdir(data_dir_bonafide)))]
    
    # save the spoof file path
    test_audio_paths.extend([os.path.join(data_dir_spoof, filename) for filename in os.listdir(data_dir_spoof)])
    test_labels.extend([1 for _ in range(len(os.listdir(data_dir_spoof)))])
    
    # Load the mean and std for testing
    mean = torch.load('training_result_three_features_directly/mean.pt')
    std = torch.load('training_result_three_features_directly/std.pt')
    # Dataset parameters: (self, all filenames, all labels, transform)
    test_dataloader = torch.utils.data.DataLoader(AudioDataset_ver2(test_audio_paths, test_labels, mean, std),
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
        for _, (spec, label) in enumerate(test_dataloader):
            # Ensure inputs are float and labels are long
            spec, label = spec.to(device).float(), label.to(device).long()
            output = model(spec)
            loss = criterion(output, label)
            test_loss += loss.item() * spec.size(0)

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
    model = CNN_model12()
    # Move model to device
    model.to(device)
    state_dict = torch.load('training_result_three_features_directly/model_12_CH_ver1_normalize.pth')
    model.load_state_dict(state_dict)
    # set loss function
    criterion = nn.CrossEntropyLoss()

    # Load audios from a Folder
    audio_folder_path_bonafide = r"D:\clone_audio\chinese_audio_dataset_ver3\bonafide\test_audio"
    audio_folder_path_spoof = r"D:\clone_audio\chinese_audio_dataset_ver3\spoof\test_audio"
    
    # record the audio_name
    dict1["Audio_name"] = [filename for filename in os.listdir(audio_folder_path_bonafide)]
    dict1["Audio_name"].extend([filename for filename in os.listdir(audio_folder_path_spoof)])
    # Load audios into test_dataloader
    test_dataloader = get_test_dataloader(audio_folder_path_bonafide, audio_folder_path_spoof)
    print(f"Loaded test data from {audio_folder_path_bonafide}\n and {audio_folder_path_spoof}.")

    # Start testing
    print("Testing...")
    test_model(model, test_dataloader, criterion)
    
    # turn dict1 into csv file
    output_df = pd.DataFrame(dict1)
    csv_file_path = f"training_result_three_features_directly/testing_result12_CH_ver1.csv"
    output_df.to_csv(csv_file_path, index=False)
    print(f"Data has been successfully exported to {csv_file_path}")