# SegNets

Just a showcase of 2 different architecture I have implemented for medical image segmentation.

## MyUNet

This is a variation of the original UNet encoder-decoder architecture, the two main differences are:
- The use of residual skip connections where UNet uses dense ones.
- The implementation of a multi-resolution outputs where one image is downscaled from the original input and processed in parallel by the network. The final output is a weighted sum of the two outputs where the weights are learned by the network.

## MySegNet

This architecture does not use a encoder-decoder structure but uses a series of dilated convolution blocks (inspired by FullNet) with residual connections, to increase the receptive field.