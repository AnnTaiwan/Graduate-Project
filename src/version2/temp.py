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

# define CNN model
class CNN_model5_small(nn.Module):
    def __init__(self):
        super(CNN_model5_small, self).__init__()
        self.input_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 3, 5, stride=1, padding=2), # kernel = 5*5
                nn.ReLU(),
                nn.BatchNorm2d(3), # 添加批次正規化層
                nn.MaxPool2d(2, stride=2) 
            )
        ])
        
        conv_filters = [12,16,12,8]
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 12, 1),
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
        
        # self.class_layers = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Dropout(0.2),
        #         # Flatten layers
        #         nn.Linear(8, 2),       
        #         # nn.ReLU(),
        #         # nn.BatchNorm1d(2),  # 添加批次正規化層
        #         # nn.Dropout(0.2),
        #         # nn.Linear(2, 2) 
        #     )
        # ])
        
    def forward(self, x):
        for layer in self.input_layers:
            x = layer(x)
        for layer in self.conv_layers:
            x = layer(x)
        # x = x.view(-1, 8)
        # for layer in self.class_layers:
        #     x = layer(x)
        return x
    
if __name__ == "__main__":
# 決定要在CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Train on {device}.")
# set up a model , turn model into cuda
    model = CNN_model5_small().to(device)

    # set loss function
    criterion = nn.CrossEntropyLoss()
    # set optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all cnn parameters

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    # Print the model summary
    summary(model, (3, 128, 128)) # Input size: (channels, height, width)
    