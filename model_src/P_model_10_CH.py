'''
Directly input mel_spec data into model
'''
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from observe_audio_function_ver3 import SAMPLE_RATE, AUDIO_LEN, get_mel_spectrogram, load_audio
import random
import os
from torchsummary import summary
import time
# Hyper Parameters
LR = 0.001
batch_size_train = 50
batch_size_valid = 50
NUM_EPOCHS = 25

# when training, do data augmentation on spoof data
def generate_dataset(audio_folder_path_bonafide, audio_folder_path_spoof, batch_size=20, train_or_valid = True):
    # load bonafide audio
    audio_paths = [os.path.join(audio_folder_path_bonafide, filename) for filename in os.listdir(audio_folder_path_bonafide)]

    # Load Labels
    # spoof is 1, bonafide is 0, see teh first character
    labels = [0 for _ in range(len(os.listdir(audio_folder_path_bonafide)))]
    
    # Load audio spec information(128, 216)
    list_spec = [] # record the spec data
    for audiopath in audio_paths:
        audio, _ = load_audio(audiopath, SAMPLE_RATE)
        # padding 0
        if len(audio) < AUDIO_LEN:
            audio = np.pad(audio, (0, AUDIO_LEN - len(audio)), 'constant') # padding zero
        else:
            audio = audio[:AUDIO_LEN]
        spec = get_mel_spectrogram(audio)
        list_spec.append(spec)

    # load spoof audio
    audio_paths = [os.path.join(audio_folder_path_spoof, filename) for filename in os.listdir(audio_folder_path_spoof)]

    # Load Labels
    # spoof is 1, bonafide is 0, see teh first character
    labels.extend([1 for _ in range(len(os.listdir(audio_folder_path_spoof)))])
    
    # Load audio spec information(128, 216)
    for audiopath in audio_paths:
        audio, _ = load_audio(audiopath, SAMPLE_RATE)
        # padding 0
        if len(audio) < AUDIO_LEN:
            audio = np.pad(audio, (0, AUDIO_LEN - len(audio)), 'constant') # padding zero
        else:
            audio = audio[:AUDIO_LEN]
        spec = get_mel_spectrogram(audio)
        list_spec.append(spec)

    # Convert list of spectrograms to a 4D tensor (batch_size, channels, height, width)
    list_spec = np.array(list_spec)  # Convert list to a numpy array
    list_spec = np.expand_dims(list_spec, axis=1)  # Add a channel dimension, assuming channel first format (batch_size, 1, height, width)
    # list_spec[0].shape is (1, 128, 216)
    # print(list_spec[0].shape)
    # Create TensorDataset
    dataset = torch.utils.data.TensorDataset(torch.tensor(list_spec, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    
    # Create DataLoader for train and validation sets
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train_or_valid, pin_memory=True)

    return data_loader
    
# define CNN model
class CNN_model10(nn.Module):
    def __init__(self):
        super(CNN_model10, self).__init__()
        self.input_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 8, 5, stride=1), # kernel = 5*5
                nn.ReLU(),
                nn.BatchNorm2d(8), # 添加批次正規化層，對同一channel作正規化
                nn.MaxPool2d(2, stride=2) 
            )
        ])
        
        conv_filters = [12,30,16,8] 
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(8, 12, 1),
                nn.ReLU(),
                nn.BatchNorm2d(12)
            ),
            nn.Sequential(
                nn.Conv2d(12, 12, 3),
                nn.ReLU(),
                nn.BatchNorm2d(12)
            ),
            nn.MaxPool2d(2, stride=2)
        ])
        for i in range(1, len(conv_filters)):
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(conv_filters[i-1], conv_filters[i], 1),
                nn.ReLU(),
                nn.BatchNorm2d(conv_filters[i])
            ))
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(conv_filters[i], conv_filters[i], 3),
                nn.ReLU(),
                nn.BatchNorm2d(conv_filters[i])
            )
            )
            self.conv_layers.append(
                nn.MaxPool2d(2, stride=2)
            )
        # final layer output above is (8, 2, 4) 93312
        self.class_layers = nn.ModuleList([
            nn.Sequential(
                # Flatten layers
                nn.Linear(8*2*4, 2),       
            )
        ])
        
    def forward(self, x):
        for layer in self.input_layers:
            x = layer(x)
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(-1, 8*2*4)
        for layer in self.class_layers:
            x = layer(x)
        return x  

# 訓練模型
def training(model):
    # 把結果寫入檔案
    file = open("training_result_mel_spec_data_directly/training_detail_model10_CH.txt", "w")
    # 紀錄最大驗證集準確率
    max_accuracy = 0

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()

        model.train() # 訓練模式
        train_loss = 0.0
        total_train = 0
        correct_train = 0
        for _, (image, label) in  enumerate(train_dataloader):
            # move tensors to GPU if CUDA is available
            image, label = image.to(device), label.to(device)
            
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(image)
            # calculate the batch loss
            loss = criterion(outputs, label)
            
            loss.backward()
            # Update the parameters
            optimizer.step()
            
            # update training loss
            train_loss += loss.item()*image.size(0)
            
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)

            total_train += label.size(0)
            correct_train += (predicted == label).sum().item()
          
        model.eval() # 改變成測試模式
        valid_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_pred = []
        all_label = []
        with torch.no_grad():
            for _, (image, label) in enumerate(valid_dataloader):
                # move tensors to GPU if CUDA is available
                image, label = image.to(device), label.to(device)

                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(image)
                # calculate the batch loss
                loss =criterion(output, label)
                # update training loss
                valid_loss += loss.item()*image.size(0)

                probs = torch.nn.functional.softmax(output, dim=1)
                _, predicted = torch.max(probs, 1)

                # Extract the probabilities for class 1 (positive class)
                probs_class_1 = np.array(probs.cpu())[:, 1] # for draw the roc

                all_probs.extend(probs_class_1)
                all_pred.extend(predicted.cpu().numpy())
                all_label.extend(label.cpu().numpy())
                total += label.size(0)
                correct += (predicted == label).sum().item()

            # 計算每個樣本的平均損失
            train_loss = train_loss / len(train_dataloader.dataset)
            valid_loss = valid_loss / len(valid_dataloader.dataset)
            Total_training_loss.append(train_loss)
            Total_validation_loss.append(valid_loss)
            
        # 計算準確率
        accuracy_train = 100 * correct_train / total_train
        accuracy_valid = 100 * correct / total
        Total_training_accuracy.append(accuracy_train)
        Total_validation_accuracy.append(accuracy_valid)

        if(accuracy_valid > max_accuracy):
            max_accuracy = accuracy_valid
            save_parameters = True
            if save_parameters:
                path = 'training_result_mel_spec_data_directly/model_10_CH_ver1.pth'
                torch.save(model.state_dict(), path)
                print(f"====Save parameters in {path}====")
                file.write(f"====Save parameters in {path}====\n")

        print(f'Epoch [{epoch+1}/{NUM_EPOCHS:d}], Train Loss: {train_loss:.4f}, Train Accuracy: {accuracy_train:.2f}%, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {accuracy_valid:.2f}%')
        file.write(f'Epoch [{epoch+1}/{NUM_EPOCHS:d}], Train Loss: {train_loss:.4f}, Train Accuracy: {accuracy_train:.2f}%, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {accuracy_valid:.2f}%\n')
        # 計算此epoch花的時間
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS:d}] took {epoch_time} seconds.")
        file.write(f"Epoch [{epoch+1}/{NUM_EPOCHS:d}] took {epoch_time} seconds.\n")

    print(f"\nMax accuracy: {max_accuracy}")
    file.write(f"\nMax accuracy: {max_accuracy}\n")
    file.close()  # 寫完後關閉檔案
    # 計算 AUROC
    roc_auc = roc_auc_score(all_label, all_probs)
    
    # 繪製最後一個epoch的ROC和confusion matrix
    # 繪製 ROC 曲線
    fpr, tpr, _ = roc_curve(all_label, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label='AUROC = {:.2f}'.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')  
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig("training_result_mel_spec_data_directly/CH_ROC10.png") 

    # confusion_matrix
    plt.figure()
    cm = confusion_matrix(all_label, all_pred)
    sns.heatmap(cm, annot=True)
    plt.savefig("training_result_mel_spec_data_directly/CH_Confusion_matrix10.png") 

def plt_loss_accuracy_fig(Total_training_loss, Total_validation_loss, Total_training_accuracy, Total_validation_accuracy):
    # visualization the loss and accuracy
    plt.figure()
    plt.plot(range(NUM_EPOCHS), Total_training_loss, 'b-', label='Training_loss')
    plt.plot(range(NUM_EPOCHS), Total_validation_loss, 'g-', label='validation_loss')
    plt.title('Training & Validation loss')
    plt.xlabel('No. of epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("training_result_mel_spec_data_directly/CH_Loss10png") 

    plt.figure()
    plt.plot(range(NUM_EPOCHS), Total_training_accuracy, 'r-', label='Training_accuracy')
    plt.plot(range(NUM_EPOCHS), Total_validation_accuracy, 'y-', label='Validation_accuracy')
    plt.title('Training & Validation accuracy')
    plt.xlabel('No. of epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("training_result_mel_spec_data_directly/CH_Accuracy10.png") 


# Start training
if __name__ == "__main__":
    # 決定要在CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Train on {device}.")
    
    # set up a model , turn model into cuda
    model = CNN_model10().to(device)
    '''
    # load P_model7 pth
    pth_path = '/workspace/model_zoo/model_API2_model7/float/model_7.pth'
    state_dict = torch.load(pth_path)
    model.load_state_dict(state_dict)
    print(f"Load model pth from {pth_path}")
    '''
    # Load audio from a Folder
    audio_folder_path_bonafide = r"D:\clone_audio\chinese_audio_dataset_ver3\bonafide\train_audio_with20db_timestretch_thchs30"
    audio_folder_path_spoof = r"D:\clone_audio\chinese_audio_dataset_ver3\spoof\train_audio_with20db_timestretch_suno_gl"
    print(f"Loading train bonafide data from {audio_folder_path_bonafide}\n and {audio_folder_path_spoof}...")
    train_dataloader = generate_dataset(audio_folder_path_bonafide, audio_folder_path_spoof, batch_size = batch_size_train, train_or_valid = True)
    
    # Load audio from a Folder
    audio_folder_path_bonafide = r"D:\clone_audio\chinese_audio_dataset_ver3\bonafide\val_audio" 
    audio_folder_path_spoof = r"D:\clone_audio\chinese_audio_dataset_ver3\spoof\val_audio"
    print(f"Loading validation data from {audio_folder_path_bonafide}\n and {audio_folder_path_spoof}...")
    valid_dataloader = generate_dataset(audio_folder_path_bonafide, audio_folder_path_spoof, batch_size = batch_size_valid, train_or_valid = False)
    print(f"Finish loading all the data.")
    
    # set loss function
    criterion = nn.CrossEntropyLoss()
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    # Print the model summary
    summary(model, (1, 128, 216)) # Input size: (channels, height, width)
  
    # 初始時間
    start_time = time.time()
    # store loss and acc data
    Total_training_loss = []
    Total_training_accuracy = []
    Total_validation_loss = []
    Total_validation_accuracy = []
   
    print("Start training....")
    # Start training
    training(model)

    # 計算總時間
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time} seconds")

    # save the fig of the loss and accuracy
    plt_loss_accuracy_fig(Total_training_loss, Total_validation_loss, Total_training_accuracy, Total_validation_accuracy)
    
