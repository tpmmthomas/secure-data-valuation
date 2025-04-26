import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class LeNet(nn.Module):
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

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class SVM(nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.flatten = nn.Flatten()
        # For CIFAR-10 images: 3 x 32 x 32 = 3072
        self.fc = nn.Linear(3072, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
class MobileNetV2(nn.Module):
     def __init__(self):
         super(MobileNetV2, self).__init__()
         self.model = models.mobilenet_v2(pretrained=False)
         # Adjust the classifier for 10 classes
         self.model.classifier[1] = nn.Linear(self.model.last_channel, 10)
 
     def forward(self, x):
         return self.model(x)


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # Feature extractor layers
        self.conv1    = nn.Conv2d(3,   64,  kernel_size=3, padding=1, stride=1)
        self.relu1    = nn.ReLU(inplace=True)
        self.pool1    = nn.MaxPool2d(kernel_size=2, stride=2)   # 32×32 → 16×16

        self.conv2    = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.relu2    = nn.ReLU(inplace=True)
        self.pool2    = nn.MaxPool2d(kernel_size=2, stride=2)   # 16×16 → 8×8

        self.conv3    = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu3    = nn.ReLU(inplace=True)

        self.conv4    = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu4    = nn.ReLU(inplace=True)

        self.conv5    = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5    = nn.ReLU(inplace=True)
        self.pool3    = nn.MaxPool2d(kernel_size=2, stride=2)   # 8×8 → 4×4

        # Classifier layers
        self.dropout1 = nn.Dropout()
        self.fc1      = nn.Linear(256 * 4 * 4, 256)
        self.relu6    = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout()
        self.fc2      = nn.Linear(256, 10)

    def forward(self, x):
        # Feature extractor
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Classifier
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def split_1(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        return x

    def split_2(self, x):
        x = self.split_1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        return x

    def split_3(self, x):
        x = self.split_2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        return x

    def split_4(self, x):
        x = self.split_3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        return x

    def split_5(self, x):
        x = self.split_4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool3(x)
        return x

    def split_6(self, x):
        x = self.split_5(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Classifier
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2,2,2,2], num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        # For CIFAR-10, use a 3x3 conv with stride 1 (no maxpooling here)
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        # No initial maxpool layer

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)  # split_2
        x = self.layer2(x)  # split_3
        x = self.layer3(x)  # split_4
        x = self.layer4(x)  # split_5
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def split_1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

    def split_2(self, x):
        x = self.split_1(x)
        x = self.layer1(x)
        return x

    def split_3(self, x):
        x = self.split_2(x)
        x = self.layer2(x)
        return x

    def split_4(self, x):
        x = self.split_3(x)
        x = self.layer3(x)
        return x

    def split_5(self, x):
        x = self.split_4(x)
        x = self.layer4(x)
        return x

    def split_6(self, x):
        x = self.split_5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    
INPUT_SIZE = (1, 3, 32, 32)