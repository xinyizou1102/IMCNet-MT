import torch.nn as nn
from torchvision.models import resnet


class ResNet101Backbone(nn.Module):
    def __init__(self):
        super(ResNet101Backbone, self).__init__()

        ResNet = resnet.resnet101(pretrained=True)
        self.conv1 = ResNet.conv1
        self.bn1 = ResNet.bn1
        self.relu = ResNet.relu
        self.maxpool = ResNet.maxpool

        self.layer1 = ResNet.layer1  # 1/4, 256
        self.layer2 = ResNet.layer2  # 1/8, 512
        self.layer3 = ResNet.layer3  # 1/16, 1024
        self.layer4 = ResNet.layer4  # 1/32, 2048

    def forward(self, x):
        output_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        output_list.append(x)
        x = self.layer2(x)
        output_list.append(x)
        x = self.layer3(x)
        output_list.append(x)
        x = self.layer4(x)
        output_list.append(x)
        return output_list


class ResNet50Backbone(nn.Module):
    def __init__(self):
        super(ResNet50Backbone, self).__init__()

        ResNet = resnet.resnet50(pretrained=True)
        self.conv1 = ResNet.conv1
        self.bn1 = ResNet.bn1
        self.relu = ResNet.relu
        self.maxpool = ResNet.maxpool

        self.layer1 = ResNet.layer1  # 1/4, 256
        self.layer2 = ResNet.layer2  # 1/8, 512
        self.layer3 = ResNet.layer3  # 1/16, 1024
        self.layer4 = ResNet.layer4  # 1/32, 2048

    def forward(self, x):
        output_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        output_list.append(x)
        x = self.layer2(x)
        output_list.append(x)
        x = self.layer3(x)
        output_list.append(x)
        x = self.layer4(x)
        output_list.append(x)
        return output_list
