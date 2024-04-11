import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class _ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        hidden_channels,
        kernel_size=2,
        padding="same",
    ):
        super(_ConvBlock, self).__init__()

        layers = []
        for i, j in zip(
            [in_channel, *hidden_channels], [*hidden_channels, out_channel]
        ):
            layers.append(nn.Conv2d(i, j, kernel_size, padding=padding))
            layers.append(nn.BatchNorm2d(i))
            layers.append(nn.ReLU(inplace=True))

        self.convs = nn.Sequential(*layers)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # 通过卷积层和激活函数序列
        x = self.convs(x)
        # 通过池化层
        x = self.pooling(x)
        return x


class Net(nn.Module):
    def __init__(self, in_channel, hidden_channels, input_dims, hidden_dims, num_class):
        super(Net, self).__init__()
        # 定义ConvBlock层
        hidden_channels = np.array(hidden_channels)
        self.conv_blocks = nn.ModuleList(
            [
                _ConvBlock(i, j, h)
                for i, j, h in zip(
                    [in_channel, *hidden_channels[:-1, -1]],
                    hidden_channels[:, -1],
                    hidden_channels[:, :-1],
                )
            ]
        )
        # 定义全连接层
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(i, j)
                for i, j in zip([int(input_dims), *hidden_dims[:-1]], hidden_dims)
            ]
        )
        self.classifier = nn.Linear(hidden_dims[-1], num_class)

    def forward(self, x):
        # 通过ConvBlock层
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        # 展平特征图
        x = x.view(x.size(0), -1)  # 展平所有维度
        # 通过全连接层
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        # 通过分类器层
        x = self.classifier(x)
        return x


# 示例使用
in_channel = 3
hidden_channels = [[6, 12], [24, 12], [6, 3]]
input_dims = hidden_channels[-1][-1] * (32 / (2 ** len(hidden_channels))) ** 2
hidden_dims = [64, 32, 16]
num_class = 10
x = torch.zeros(16, 3, 32, 32)

model = Net(in_channel, hidden_channels, input_dims, hidden_dims, num_class)
scores = model(x)
print(scores.shape)
