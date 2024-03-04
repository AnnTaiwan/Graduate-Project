from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

import os
from torchsummary import summary
import time
# Hyper Parameters
LR = 0.02
batch_size_train = 10
batch_size_valid = 5
# n_iters = 10000
NUM_EPOCHS = 5

IMAGE_SIZE = 128
transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # Original Size: (640, 480)
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

# 設定dataset，包含設定transformer、分成70% / 30%，圖片轉換處理完直接存到list，(沒有做成pytorch中的dataset)
def get_train_test_dataloader(image_folder_path):
    image_paths = [os.path.join(image_folder_path, filename) for filename in os.listdir(image_folder_path)]

    # Load Labels
    # Assuming 'train_df' has columns 'filename' and 'target'
    # skip the "spec_" and ".png"
    labels = [train_df[train_df["filename"] == os.path.basename(path)[5:-4]]["target"].values[0] for path in image_paths]

    # Apply Transformations
    # get the cropped image
    # Convert the NumPy array back to an image
    images = [transform(Image.fromarray(remove_margin(path)[1].astype('uint8')).convert('RGB')) for path in image_paths]
    # images = [transform(Image.open(path).convert('RGB')) for path in image_paths]

    # Create TensorDataset
    train_data = torch.utils.data.TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.long))

    # Define Train and Validation Sizes
    train_size = int(0.7 * len(train_data))
    valid_size = len(train_data) - train_size

    # Split into train and validation sets
    train_dataset, valid_dataset = torch.utils.data.random_split(train_data, [train_size, valid_size])

    # Create DataLoader for train and validation sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size_valid, shuffle=False)

    return train_loader, valid_loader

# define CNN model
class CNN_model3(nn.Module):
    def __init__(self):
        super(CNN_model3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3) 
        self.bn1 = nn.BatchNorm2d(32)  # 添加批次正規化層
        self.pool1 = nn.MaxPool2d(2, 2) 
        self.conv2 = nn.Conv2d(32, 64, 4) 
        self.bn2 = nn.BatchNorm2d(64)  # 添加批次正規化層
        self.pool2 = nn.MaxPool2d(2, 2) 
        self.fc1 = nn.Linear(64 * 30 * 30, 512) 
        self.fc2 = nn.Linear(512, 2) 
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))) 
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x))) 
        x = self.pool2(x)
        x = self.dropout(x)
        
        x = x.view(-1, 64 * 30 * 30) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
# 訓練模型
def training(model):
    # 把結果寫入檔案
    file = open(r"D:\graduate_project\src\version2\training_detail_model4.txt", "w")
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
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS:d}], Train Loss: {train_loss:.4f}, Train Accuracy: {accuracy_train:.2f}%, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {accuracy_valid:.2f}%')
        file.write(f'Epoch [{epoch+1}/{NUM_EPOCHS:d}], Train Loss: {train_loss:.4f}, Train Accuracy: {accuracy_train:.2f}%, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {accuracy_valid:.2f}%\n')
        # 計算此epoch花的時間
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS:d}] took {epoch_time} seconds.")
        file.write(f"Epoch [{epoch+1}/{NUM_EPOCHS:d}] took {epoch_time} seconds.\n")

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
    plt.savefig(r"D:\graduate_project\src\version2\ROC4.png") 

    # confusion_matrix
    plt.figure()
    cm = confusion_matrix(all_label, all_pred)
    sns.heatmap(cm, annot=True)
    plt.savefig(r"D:\graduate_project\src\version2\Confusion_matrix4.png") 

def plt_loss_accuracy_fig(Total_training_loss, Total_validation_loss, Total_training_accuracy, Total_validation_accuracy):
    # visualization the loss and accuracy
    plt.figure()
    plt.plot(range(NUM_EPOCHS), Total_training_loss, 'b-', label='Training_loss')
    plt.plot(range(NUM_EPOCHS), Total_validation_loss, 'g-', label='validation_loss')
    plt.title('Training & Validation loss')
    plt.xlabel('No. of epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(r"D:\graduate_project\src\version2\Loss4.png") 

    plt.figure()
    plt.plot(range(NUM_EPOCHS), Total_training_accuracy, 'r-', label='Training_accuracy')
    plt.plot(range(NUM_EPOCHS), Total_validation_accuracy, 'y-', label='Validation_accuracy')
    plt.title('Training & Validation accuracy')
    plt.xlabel('No. of epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(r"D:\graduate_project\src\version2\Accuracy4.png") 


# Start training
if __name__ == "__main__":
    # 決定要在CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Train on {device}.")
    # 讀取LA_train_info
    train_df = pd.read_csv("train_info.csv")
    # Load Images from a Folder
    image_folder_path = r"D:/graduate_project/src/spec_LATrain_audio_shuffle1_NOT_preprocessing"
    train_dataloader, valid_dataloader = get_train_test_dataloader(image_folder_path)
    print("Loaded data.")

    # set up a model , turn model into cuda
    model = CNN_model3().to(device)

    # set loss function
    criterion = nn.CrossEntropyLoss()
    # set optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)   # optimize all cnn parameters

    # Print the model summary
    summary(model, (3, IMAGE_SIZE, IMAGE_SIZE)) # Input size: (channels, height, width)

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

    save_parameters = True
    if save_parameters:
        torch.save(model.state_dict(), 'model_4.h5')
