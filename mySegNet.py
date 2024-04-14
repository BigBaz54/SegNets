import torch
import torch.nn as nn
from models.CNNBaseModel import CNNBaseModel

class MySegNet(CNNBaseModel):
    """
    Class that implements a brand new segmentation CNN inspired by FullNet
    """
    def __init__(self, num_classes=4, init_weights=True):
        """
        Builds MySegNet model.
        Args:
            num_classes (int): number of classes. default 4
            init_weights (bool): when true uses _initialize_weights function to initialize network's weights.
        """
        super().__init__(num_classes, init_weights)

        # initial convolutional block that reduces the size of the input
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        ) # 256x256 -> 128x128 -> 64x64

        # dilated convolutional blocks to increase receptive field
        self.dilated_conv1 = self.dilated_conv_block(128, 256, dilation=2)
        self.one_by_one1 = self.one_by_one_block(128, 256)

        self.dilated_conv2 = self.dilated_conv_block(256, 512, dilation=4)
        self.one_by_one2 = self.one_by_one_block(256, 512)

        self.dilated_conv3 = self.dilated_conv_block(512, 1024, dilation=8)
        self.one_by_one3 = self.one_by_one_block(512, 1024)

        # final convolutional block that increases back the size of the input
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, num_classes, kernel_size=1),
        ) # 64x64 -> 128x128 -> 256x256

    def dilated_conv_block(self, in_channels, out_channels, dilation):
        """
        Dilated convolutional block with one dilated convolution and batch normalization.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
    
    def one_by_one_block(self, in_channels, out_channels):
        """
        Convolutional block with one 1x1 convolution and batch normalization.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """
        x = self.initial_conv(x)
        
        x = self.dilated_conv1(x) + self.one_by_one1(x)
        x = self.dilated_conv2(x) + self.one_by_one2(x)
        x = self.dilated_conv3(x) + self.one_by_one3(x)

        x = self.final_conv(x)

        return x
