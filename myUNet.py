import torch
import torch.nn as nn
from models.CNNBaseModel import CNNBaseModel
from torch.nn import functional as F


class MyUNet(CNNBaseModel):
    """
     Class that implements a brand new UNet segmentation network
    """

    def __init__(self, num_classes=4, init_weights=True):
        """
        Builds MyUNet  model.
        Args:
            num_classes(int): number of classes. default 10(cifar10 or svhn)
            init_weights(bool): when true uses _initialize_weights function to initialize
            network's weights.
        """
        super().__init__(num_classes, init_weights)

        in_channels = 1

        self.first_conv = self.conv_block(in_channels, 64) # 256x256 and 128x128

        # encoder
        self.encoder1 = self.down_block(64, 128) # 128x128 and 64x64
        self.encoder2 = self.down_block(128, 256) # 64x64 and 32x32
        self.encoder3 = self.down_block(256, 512) # 32x32 and 16x16
        self.encoder4 = self.down_block(512, 1024) # 16x16 and 8x8

        # middle
        self.middle = self.middle_block(1024) # 16x16 and 8x8

        # decoder
        self.decoder4 = self.up_block(1024, 512) # 32x32 and 16x16
        self.decoder3 = self.up_block(512, 256) # 64x64 and 32x32
        self.decoder2 = self.up_block(256, 128) # 128x128 and 64x64
        self.decoder1 = self.up_block(128, 64) # 256x256 and 128x128

        self.last_conv1 = nn.Conv2d(64, 64, kernel_size=1) # 256x256 and 128x128
        self.last_conv2 = nn.Conv2d(64, num_classes, kernel_size=1) # 256x256 and 128x128

        # upscaling for low resolution branch
        self.low_res_upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # 128x128 to 256x256

        # weights for the two branches
        self.weights = nn.Parameter(torch.tensor([0.5, 0.5], requires_grad=True))

    def down_block(self, in_channels, out_channels, kernel_size=3):
        """
        Block that performs downsampling by reducing the spatial dimensions by half.
        Args:
            in_channels(int): number of input channels
            out_channels(int): number of output channels
            kernel_size(int): size of the convolutional kernel
        """
        return nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def up_block(self, in_channels, out_channels, kernel_size=3):
        """
        Block that performs upsampling by increasing the spatial dimensions by double.
        Args:
            in_channels(int): number of input channels
            out_channels(int): number of output channels
            kernel_size(int): size of the convolutional kernel
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
    
    def middle_block(self, channels):
        """
        Middle block of the network.
        Args:
            channels(int): number of input and output channels
        """
        return nn.Sequential(
            self.bottleneck_block(channels, 4),
            self.bottleneck_block(channels, 4),
        )
    
    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """

        low_res_x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)

        # first convolution
        first = self.first_conv(x)
        low_res_first = self.first_conv(low_res_x)

        # high resolution branch encoding
        encoder1 = self.encoder1(first)
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)

        # low resolution branch encoding
        low_res_encoder1 = self.encoder1(low_res_first)
        low_res_encoder2 = self.encoder2(low_res_encoder1)
        low_res_encoder3 = self.encoder3(low_res_encoder2)
        low_res_encoder4 = self.encoder4(low_res_encoder3)

        # middle
        middle = self.middle(encoder4)
        low_res_middle = self.middle(low_res_encoder4)

        # high resolution branch decoding
        decoder4 = self.decoder4(middle + encoder4)
        decoder3 = self.decoder3(decoder4 + encoder3)
        decoder2 = self.decoder2(decoder3 + encoder2)
        decoder1 = self.decoder1(decoder2 + encoder1)

        # low resolution branch decoding
        low_res_decoder4 = self.decoder4(low_res_middle + low_res_encoder4)
        low_res_decoder3 = self.decoder3(low_res_decoder4 + low_res_encoder3)
        low_res_decoder2 = self.decoder2(low_res_decoder3 + low_res_encoder2)
        low_res_decoder1 = self.decoder1(low_res_decoder2 + low_res_encoder1)

        # last convolutions
        final = self.last_conv1(decoder1)
        final = self.last_conv2(final)
        low_res_final = self.last_conv1(low_res_decoder1)
        low_res_final = self.last_conv2(low_res_final)

        # upscaling low resolution branch
        low_res_output = self.low_res_upscale(low_res_final)

        # final convolutions
        output1 = final
        output2 = low_res_output
        output = self.weights[0] * output1 + self.weights[1] * output2

        return output
    
    def bottleneck_block(self, in_channels, downsample=4):
        """
        Fixed bottleneck block with same convolutions.
        Args:
            in_channels(int): number of input channels
            downsample(int): number of channels to downsample
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels // downsample, kernel_size=1, stride=1,
                       bias=False),
            nn.Conv2d(in_channels=in_channels // downsample, out_channels=in_channels // downsample,
                       kernel_size=3, stride=1, bias=False, padding=1),
            nn.Conv2d(in_channels=in_channels // downsample, out_channels=in_channels, kernel_size=1, stride=1,
                       bias=False)
        )
    
    def conv_block(self, in_channels, out_channels, kernel_size=3):
        """
        Classic convolutional block with two convolutions and batch normalization.
        """
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        return block