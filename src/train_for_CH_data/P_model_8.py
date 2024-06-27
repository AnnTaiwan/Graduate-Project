'''
Compare to P-model_7,
deleting dropout function
Here train the chinese data
input size become 256 * 256 
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

import random
import os
from torchsummary import summary
import time
# Hyper Parameters
LR = 0.02
batch_size_train = 20
batch_size_valid = 20
# n_iters = 10000
NUM_EPOCHS = 30

IMAGE_SIZE = 256
transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # Original Size: (640, 480)
        transforms.ToTensor()
    ])
# do data augmentation on spoof data
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomAffine(degrees=0, translate=(0.2, 0), scale=None, shear=0),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
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
def get_train_test_dataloader(bonafide_dir, spoof_dir):
    # list all the paths
    real_paths = [os.path.join(bonafide_dir, filename) for filename in os.listdir(bonafide_dir)]
    fake_paths = [os.path.join(spoof_dir, filename) for filename in os.listdir(spoof_dir)]
    # setting size
    '''
    Train:
        real : 800
        fake : 30? + 30? (aug)
    Val:
        real : 100
        fake : 100
    '''
    real_train_num = 800
    real_val_num = 100
    fake_val_num = 100
    fake_train_num = len(fake_paths) - fake_val_num

    # get 800 bonafide images
    real_indices = list(range(len(real_paths)))
    random.shuffle(real_indices)
    real_train_indices = real_indices[:real_train_num]
    real_val_indices = real_indices[real_train_num:real_train_num+real_val_num]

    # get all spoof images
    fake_indices = list(range(len(fake_paths)))
    random.shuffle(fake_indices)
    fake_train_indices = fake_indices[fake_val_num:]
    fake_val_indices = fake_indices[:fake_val_num]
    # Load Labels
    real_train_label = [0 for i in range(real_train_num)]
    fake_train_label = [1 for i in range(fake_train_num*2)]

    real_val_label = [0 for i in range(real_val_num)]
    fake_val_label = [1 for i in range(fake_val_num)]   
    
    # load images
    image_real_train_paths = [real_paths[i] for i in real_train_indices]
    image_real_val_paths = [real_paths[i] for i in real_val_indices]
    image_fake_train_paths = [fake_paths[i] for i in fake_train_indices]
    image_fake_val_paths = [fake_paths[i] for i in fake_val_indices]

    # remove margin and do some transform and normalize it 
    image_real_train = [transform(Image.fromarray(remove_margin(path)[1].astype('uint8')).convert('RGB')) / 255.0 for path in image_real_train_paths]
    image_real_val = [transform(Image.fromarray(remove_margin(path)[1].astype('uint8')).convert('RGB')) / 255.0 for path in image_real_val_paths]
    image_fake_train = [transform(Image.fromarray(remove_margin(path)[1].astype('uint8')).convert('RGB')) / 255.0 for path in image_fake_train_paths]
    image_fake_val = [transform(Image.fromarray(remove_margin(path)[1].astype('uint8')).convert('RGB')) / 255.0 for path in image_fake_val_paths]
    # create data augmentation data
    image_fake_train_aug = [augment_transform(Image.fromarray(remove_margin(path)[1].astype('uint8')).convert('RGB')) / 255.0  for path in image_fake_train_paths]
    # concat this two train fake data
    image_fake_train += image_fake_train_aug
    # shuffle it
    random.shuffle(image_fake_train)
    # concat all train and val data
    train_images = image_real_train + image_fake_train
    train_labels = real_train_label + fake_train_label
    val_images = image_real_val + image_fake_val
    val_labels = real_val_label + fake_val_label

    # Create TensorDataset
    train_dataset = torch.utils.data.TensorDataset(torch.stack(train_images), torch.tensor(train_labels, dtype=torch.long))
    valid_dataset = torch.utils.data.TensorDataset(torch.stack(val_images), torch.tensor(val_labels, dtype=torch.long))
    
    # Create DataLoader for train and validation sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size_valid, shuffle=False, pin_memory=True)
   
    return train_loader, valid_loader
    

# define CNN model
class CNN_model8(nn.Module):
    def __init__(self):
        super(CNN_model8, self).__init__()
        self.input_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 8, 5, stride=1), # kernel = 5*5
                nn.ReLU(),
                nn.BatchNorm2d(8), # 添加批次正規化層，對同一channel作正規化
                nn.MaxPool2d(2, stride=2) 
            )
        ])
        
        conv_filters = [12,30,16,8] # [12,16,12,8]
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
        # final layer output above is (8, 108, 108) 93312
        self.class_layers = nn.ModuleList([
            nn.Sequential(
                # Flatten layers
                nn.Linear(8*6*6, 10),   
                nn.Linear(10, 2)     
            )
        ])
        
    def forward(self, x):
        for layer in self.input_layers:
            x = layer(x)
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(-1, 8*6*6)
        for layer in self.class_layers:
            x = layer(x)
        return x
  
# 訓練模型
def training(model):
    # 把結果寫入檔案
    file = open(r"D:\graduate_project\src\train_for_CH_data\CH_training_detail_model8.txt", "w")
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
                # path = 'model_CH_8.pth'
                path = 'model_CH_8_2.pth'

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
    plt.savefig(r"D:\graduate_project\src\train_for_CH_data\CH_ROC8.png") 

    # confusion_matrix
    plt.figure()
    cm = confusion_matrix(all_label, all_pred)
    sns.heatmap(cm, annot=True)
    plt.savefig(r"D:\graduate_project\src\train_for_CH_data\CH_Confusion_matrix8.png") 

def plt_loss_accuracy_fig(Total_training_loss, Total_validation_loss, Total_training_accuracy, Total_validation_accuracy):
    # visualization the loss and accuracy
    plt.figure()
    plt.plot(range(NUM_EPOCHS), Total_training_loss, 'b-', label='Training_loss')
    plt.plot(range(NUM_EPOCHS), Total_validation_loss, 'g-', label='validation_loss')
    plt.title('Training & Validation loss')
    plt.xlabel('No. of epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(r"D:\graduate_project\src\train_for_CH_data\CH_Loss8.png") 

    plt.figure()
    plt.plot(range(NUM_EPOCHS), Total_training_accuracy, 'r-', label='Training_accuracy')
    plt.plot(range(NUM_EPOCHS), Total_validation_accuracy, 'y-', label='Validation_accuracy')
    plt.title('Training & Validation accuracy')
    plt.xlabel('No. of epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(r"D:\graduate_project\src\train_for_CH_data\CH_Accuracy8.png") 


# Start training
if __name__ == "__main__":
    # 決定要在CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Train on {device}.")
    # Load Images from a Folder
    bonafide_dir = r"D:\clone_audio\chinese_audio_spec\bonafide_spec_base\part_1"
    spoof_dir = r"D:\clone_audio\chinese_audio_spec\spoof_spec_total"
    train_dataloader, valid_dataloader = get_train_test_dataloader(bonafide_dir, spoof_dir)
    print(f"Loaded data from {bonafide_dir}\nand {spoof_dir}.")
    
    # set up a model , turn model into cuda
    model = CNN_model8().to(device)

    # set loss function
    criterion = nn.CrossEntropyLoss()
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all cnn parameters

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
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
    