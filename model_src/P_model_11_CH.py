'''
1. Use crop_with_points to remove margin
2. input mel_spec, mfcc, chroma image(totally 9 channels) ,and use RGB pixel value as input feature
'''
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import  transforms

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

import os
from torchsummary import summary
import time
# Hyper Parameters
LR = 0.001
batch_size_train = 50
batch_size_valid = 50
NUM_EPOCHS = 20

IMAGE_SIZE = 128
transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # Original Size: (640, 480)
        transforms.ToTensor()
    ])



# 移除圖片周圍空白處
def crop_with_points(image_path):
    points = [(79, 57), (575, 428), (575, 57), (79, 428)]
    # Load the image
    img = Image.open(image_path)
    # original shape is 640*480
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
    cropped_img = img.crop((left, upper, right, lower)) # 79, 57, 575, 428
	# After cropping, shape is 496*369
    return cropped_img

# when training, do data augmentation on spoof data
def generate_dataset(df, batch_size=20, train_or_valid=True):
    images = []
    labels = []

    # Filter DataFrame for training or validation set
    subset_df = df[df["type"] == ("train" if train_or_valid else "val")]

    for _, row in subset_df.iterrows():
        # get each path
        mel_spec_path = row["Mel_spec_path"]
        mfcc_path = row["MFCC_path"]
        chroma_path = row["Chroma_path"]
        # get each image
        mel_spec_image = transform(crop_with_points(mel_spec_path).convert('RGB'))
        mfcc_image = transform(crop_with_points(mfcc_path).convert('RGB'))
        chroma_image = transform(crop_with_points(chroma_path).convert('RGB'))

        # Stack images along the channel dimension
        stacked_image = torch.cat([mel_spec_image, mfcc_image, chroma_image],  dim=0) 
        # print(np.array(stacked_image).shape) # (9, 128, 128)
        images.append(stacked_image)                     
        labels.append(row["Label"])
    
    # Create TensorDataset
    dataset = torch.utils.data.TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.long))

    # Create DataLoader for train and validation sets
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train_or_valid, pin_memory=True)

    return data_loader
    
# define CNN model
class CNN_model11(nn.Module):
    def __init__(self):
        super(CNN_model11, self).__init__()
        self.input_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(9, 12, 5, stride=1), # kernel = 5*5
                nn.ReLU(),
                nn.BatchNorm2d(12), # 添加批次正規化層，對同一channel作正規化
                nn.MaxPool2d(2, stride=2) 
            )
        ])
        
        conv_filters = [15,30,16,8] 
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(12, 15, 1),
                nn.ReLU(),
                nn.BatchNorm2d(15)
            ),
            nn.Sequential(
                nn.Conv2d(15, 15, 3),
                nn.ReLU(),
                nn.BatchNorm2d(15)
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
        
        self.class_layers = nn.ModuleList([
            nn.Sequential(
                # Flatten layers
                nn.Linear(8*2*2, 2)     
            )
        ])
        
    def forward(self, x):
        for layer in self.input_layers:
            x = layer(x)
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(-1, 8*2*2)
        for layer in self.class_layers:
            x = layer(x)
        return x  

# 訓練模型
def training(model):
    # 把結果寫入檔案
    file = open("training_result_three_features_image/training_detail_model11_CH.txt", "w")
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
                path = 'training_result_three_features_image/model_11_CH_ver1.pth'
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
    plt.savefig("training_result_three_features_image/CH_ROC11.png") 

    # confusion_matrix
    plt.figure()
    cm = confusion_matrix(all_label, all_pred)
    sns.heatmap(cm, annot=True)
    plt.savefig("training_result_three_features_image/CH_Confusion_matrix11.png") 

def plt_loss_accuracy_fig(Total_training_loss, Total_validation_loss, Total_training_accuracy, Total_validation_accuracy):
    # visualization the loss and accuracy
    plt.figure()
    plt.plot(range(NUM_EPOCHS), Total_training_loss, 'b-', label='Training_loss')
    plt.plot(range(NUM_EPOCHS), Total_validation_loss, 'g-', label='validation_loss')
    plt.title('Training & Validation loss')
    plt.xlabel('No. of epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("training_result_three_features_image/CH_Loss11.png") 

    plt.figure()
    plt.plot(range(NUM_EPOCHS), Total_training_accuracy, 'r-', label='Training_accuracy')
    plt.plot(range(NUM_EPOCHS), Total_validation_accuracy, 'y-', label='Validation_accuracy')
    plt.title('Training & Validation accuracy')
    plt.xlabel('No. of epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("training_result_three_features_image/CH_Accuracy11.png") 


# Start training
if __name__ == "__main__":
    # CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Train on {device}.")
    # load the info csv
    info_path = "chinese_audio_dataset_ver3_info.csv"
    info_df = pd.read_csv(info_path)
    # set up a model , turn model into cuda
    model = CNN_model11().to(device)
    
    print(f"Loading train data from {info_path}...")
    train_dataloader = generate_dataset(info_df, batch_size = batch_size_train, train_or_valid = True)
    
    print(f"Loading validation data from {info_path}...")
    valid_dataloader = generate_dataset(info_df, batch_size = batch_size_valid, train_or_valid = False)
    print(f"Finish loading all the data.")
    
    
    
    # set loss function
    criterion = nn.CrossEntropyLoss()
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    # Print the model summary
    summary(model, (9, IMAGE_SIZE, IMAGE_SIZE)) # Input size: (channels, height, width)
    
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
    
    