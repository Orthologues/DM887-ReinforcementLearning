"""
This file defines the CNN feature extractor for the BDQN algorithm.
which consists of three convolutional layers, one fully-connected layer to extract the features of the normalized and rescaled input tensor,
"""

from torch import nn
from ..utils import Config
from torch import nn, Tensor, relu

class BdqnConvNet(nn.Module):

    FEATURE_DIM = 512

    def __init__(self, input_dim, feature_dim=FEATURE_DIM, num_filters=32):
        super().__init__()
        self.init_body(input_dim, feature_dim, num_filters)
        # feed this module to GPU
        self.to(Config.DEVICE)

    def init_body(self, input_dim, output_dim, num_filters) -> None:
        """
        Desired input shape: (32, 4, 84, 84)
        Output Shape 1: (32,32,20,20)
        Output Shape 2: (32,64,9,9)
        Output Shape 3: (32,64,7,7)
        Input Shape 4 (after flattening): (32,3136) # 3136=7*7*64
        Output Shape 4: (32, 512)
        """
        # the 1th layer of the feature extractor (convolutional)
        self.conv1 = self.init_layer(nn.Conv2d(in_channels=input_dim[1], out_channels=num_filters, kernel_size=8, stride=4))
        self.bn1 = nn.BatchNorm2d(num_filters) # batch normalization for the output of conv layer 1
        # the 2th layer of the feature extractor (convolutional)
        self.conv2 = self.init_layer(nn.Conv2d(num_filters, num_filters*2, kernel_size=4, stride=2))
        self.bn2 = nn.BatchNorm2d(num_filters*2) # batch normalization for the output of conv layer 2
        # the 3th layer of the feature extractor (convolutional)
        self.conv3 = self.init_layer((nn.Conv2d(num_filters*2, num_filters*2, kernel_size=3, stride=1)))
        self.bn3 = nn.BatchNorm2d(num_filters*2) # batch normalization for the output of conv layer 3

        # the 4th layer of the feature extractor (linear)
        self.fc4 = self.init_layer(nn.Linear(7 * 7 * 64, output_dim))


    def init_layer(layer: nn.Module) -> nn.Module:
        nn.init.orthogonal_(layer.weight.data)
        nn.init.constant_(layer.bias.data, 0)
        return layer
    

    def forward(self, x: Tensor):
        x = relu(self.bn1(self.conv1(x)))
        x = relu(self.bn2(self.conv2(x)))
        x = relu(self.bn3(self.conv3(x)))
        x = x.flatten(start_dim=1)
        x = relu(self.fc4(x))
        return x
        
