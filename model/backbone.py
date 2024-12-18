"""
Author: yida
Time is: 2022/1/7 19:49
this Code: 重新实现真正的Resnet18
从encoder中提取特征 计算sdtw_loss
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 1.基础模块, 两次卷积, 然后跳跃连接相加
class BasicBlock(nn.Module):
    def __init__(self, in_channel):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = F.relu(out + identity)
        return out


# 2.下采样模块, 第一次卷积缩小尺度扩大维度 第二次卷积不变, 跳跃连接使用1*1卷积缩小尺度扩大维度 与第二次卷积相加
class DownSample(nn.Module):
    def __init__(self, in_channel):
        super(DownSample, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 2 * in_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * in_channel),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * in_channel, 2 * in_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 * in_channel),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channel, 2 * in_channel, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(2 * in_channel),

        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        identity = self.downsample(identity)
        out = F.relu(out + identity)
        return out


class ResNet18(nn.Module):
    def __init__(self, feature_dim):
        super(ResNet18, self).__init__()
        # 最初的卷积和最大池化
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 第一层
        self.layer1 = nn.Sequential(
            BasicBlock(64),
            BasicBlock(64),
        )
        # 第二层
        self.layer2 = nn.Sequential(
            DownSample(64),
            BasicBlock(128),
        )
        # 第三层
        self.layer3 = nn.Sequential(
            DownSample(128),
            BasicBlock(256),
        )
        # 第四层
        self.layer4 = nn.Sequential(
            DownSample(256),
            BasicBlock(512),
        )
        # 全局均值池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # n_class 为分类数
        self.fc = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                # nn.Dropout(p=0.01),
                nn.Linear(512, 512),
                nn.ReLU(),
                # nn.Dropout(p=0.1),
                nn.Linear(512,feature_dim),

            )
        # self.fc = nn.Sequential(
        #         nn.Linear(512, 512),
        #         nn.ReLU(),
        #         # nn.Dropout(p=0.01),
        #         # nn.Dropout(p=0.1),
        #         nn.Linear(512,feature_dim),
        #
        #     )

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        h = x.view(x.size(0), -1)
        out = self.fc(h)
        return F.normalize(out, dim=-1),F.normalize(out, dim=-1)


# if __name__ == '__main__':
#     inputs = torch.rand(10, 3, 224, 224)
#     model = ResNet18(feature_dim=128)
#     print(model)
#     outputs = model(inputs)
#     print(outputs.shape)

