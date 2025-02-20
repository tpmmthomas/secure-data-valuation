import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

class SimpleCifarCNN(nn.Module):
    def __init__(self):
        super(SimpleCifarCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class cnnNet(nn.Module):
    def __init__(self):
        super(cnnNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        # output = self.act(x)
        return output
    
class ResNetCIFAR10(nn.Module):
    def __init__(self):
        super(ResNetCIFAR10, self).__init__()
        # Load a pre-trained ResNet-18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Replace the final fully connected layer with one for 10 classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)
        # Define a softmax layer to convert logits to probabilities (for inference)
        # self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Get logits from the ResNet
        logits = self.resnet(x)
        # Compute probabilities using softmax (for inference/analysis)
        # probs = self.softmax(logits)
        return logits
    
def get_model(model):
    if model == "resnet18":
        outmodel = ResNetCIFAR10()
    elif model == "vgg16":
        outmodel = models.vgg16(pretrained=True)
    elif model == "cifarcnn":
        outmodel = SimpleCifarCNN()
    elif model == "cnn":
        outmodel = cnnNet()
    else:
        raise ValueError("Unknown model")

    if torch.cuda.is_available():
        outmodel = outmodel.cuda()
    
    return outmodel


def get_convex_model(model,random_state=42):
    if model == "logistic":
        outmodel = LogisticRegression(max_iter=1000, random_state=random_state)
    elif model == "svm":
        outmodel = SVC(random_state=random_state)
        
    return outmodel