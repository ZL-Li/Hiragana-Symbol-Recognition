# kuzu.py
# python 3.7.7
# torch 1.6.0
# torchversion 0.7.0

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # input layer(784) -> output layer(10)
        self.in_to_out = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1) # flatten tensor x's dimension from [batch_size, 1, 28, 28] to [batch_size, 1 * 28 * 28]
        output = F.log_softmax(self.in_to_out(x), dim = 1)
        return output

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # input layer(784) -> hidden layer(180) -> output layer(10)
        self.in_to_hidden = nn.Linear(28 * 28, 180)
        self.hidden_to_out = nn.Linear(180, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1) # flatten tensor x's dimension from [batch_size, 1, 28, 28] to [batch_size, 1 * 28 * 28]
        hidden_output = torch.tanh(self.in_to_hidden(x))
        output = F.log_softmax(self.hidden_to_out(hidden_output), dim = 1)
        return output

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # input layer(1, 28, 28) -> conv1 layer(16, 28, 28) -> pool1 layer(16, 14, 14) -> 
        # conv2 layer(32, 14, 14) -> pool2 layer(32, 7, 7) -> flattening to (32 * 7 * 7) ->
        # hidden layer(180) -> output layer(10)
        self.in_to_conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5, stride = 1, padding = 2)
        self.conv1_to_pool1 = nn.MaxPool2d(kernel_size = 2)
        self.pool1_to_conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 1, padding = 2)
        self.conv2_to_pool2 = nn.MaxPool2d(kernel_size = 2)
        self.pool2_to_hidden = nn.Linear(32 * 7 * 7, 600)
        self.hidden_to_out = nn.Linear(600, 10)

    def forward(self, x):
        conv1_output = F.relu(self.in_to_conv1(x))
        pool1_output = self.conv1_to_pool1(conv1_output)
        conv2_output = F.relu(self.pool1_to_conv2(pool1_output))
        pool2_output = self.conv2_to_pool2(conv2_output)

        # flatten tensor pool2_output's dimension from [batch_size, 32, 7, 7] to [batch_size, 32 * 7 * 7]
        pool2_output = pool2_output.view(pool2_output.size(0), -1)

        hidden_output = F.relu(self.pool2_to_hidden(pool2_output))
        output = F.log_softmax(self.hidden_to_out(hidden_output), dim = 1)
        return output
