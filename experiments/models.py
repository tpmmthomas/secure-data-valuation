import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class LeNet(nn.Sequential):
    """
    Adaptation of LeNet that uses ReLU activations
    """

    # network architecture:
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.act = nn.Softmax(dim=1)
        

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.act(x)
        return x

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
    elif model == "lenet":
        outmodel = LeNet()
    elif model == "cnn":
        outmodel = cnnNet()
    else:
        raise ValueError("Unknown model")

    if torch.cuda.is_available():
        outmodel = outmodel.cuda()
    
    return outmodel