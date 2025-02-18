import torch 
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet101,
    deeplabv3_resnet50,
    DeepLabV3_MobileNet_V3_Large_Weights,
    DeepLabV3_ResNet101_Weights,
    DeepLabV3_ResNet50_Weights
)
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from torchvision.models import resnet50, ResNet50_Weights



""" 
  !┌─────────────────────┐
  *│ Originial Deeplabv3 │
  !└─────────────────────┘
"""

class DeepLabV3Model:
    """Class for selecting and fine-tuning DeepLabV3 models.

    Attributes:
        model: The selected DeepLabV3 model.
    """

    def __init__(self, model_name: str, num_classes: int):
        """Initializes the DeepLabV3Model class.

        Args:
            model_name (str): The name of the DeepLabV3 model to use. Options: 'mobilenet_v3_large', 'resnet101', 'resnet50'.
            num_classes (int): The number of output classes.
        """
        self.model = self._load_model(model_name, num_classes)

    def _load_model(self, model_name: str, num_classes: int) -> torch.nn.Module:
        """Loads the specified DeepLabV3 model and modifies the last layer.

        Args:
            model_name (str): The name of the DeepLabV3 model to use.
            num_classes (int): The number of output classes.

        Returns:
            torch.nn.Module: The modified DeepLabV3 model.
        """
        if model_name == 'mobilenet_v3_large':
            weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
            model = deeplabv3_mobilenet_v3_large(weights=weights)
            in_channels = 960
        elif model_name == 'resnet101':
            weights = DeepLabV3_ResNet101_Weights.DEFAULT
            model = deeplabv3_resnet101(weights=weights)
            in_channels = 2048
        elif model_name == 'resnet50':
            weights = DeepLabV3_ResNet50_Weights.DEFAULT
            model = deeplabv3_resnet50(weights=weights)
            in_channels = 2048
        else:
            raise ValueError("Invalid model name. Choose from 'mobilenet_v3_large', 'resnet101', 'resnet50'.")

        # Modify the classifier to output the correct number of classes
        model.classifier = DeepLabHead(in_channels, num_classes)
        
        return model

    def get_model(self) -> torch.nn.Module:
        """Returns the selected DeepLabV3 model.

        Returns:
            torch.nn.Module: The DeepLabV3 model.
        """
        return self.model


""" 
  !┌────────────┐
  *│ DeepLabv3+ │
  !└────────────┘
"""


class Atrous_Convolution(nn.Module):
    """
    Compute Atrous/Dilated Convolution.
    """

    def __init__(self, input_channels: int, kernel_size: int, pad: int, dilation_rate: int, output_channels: int = 256) -> None:
        """
        Initializes the Atrous_Convolution module.

        Args:
            input_channels (int): Number of input channels.
            kernel_size (int): Size of the convolution kernel.
            pad (int): Padding size.
            dilation_rate (int): Dilation rate for the convolution.
            output_channels (int, optional): Number of output channels. Defaults to 256.
        """
        super(Atrous_Convolution, self).__init__()

        self.conv = nn.Conv2d(in_channels=input_channels,
                              out_channels=output_channels,
                              kernel_size=kernel_size, padding=pad,
                              dilation=dilation_rate, bias=False)

        self.batchnorm = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the convolution, batch normalization, and ReLU activation.
        """
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class ASSP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASSP) layer used in the encoder of DeepLabv3+.
    """

    def __init__(self, in_channles: int, out_channles: int) -> None:
        """
        Initializes the ASSP module.

        Args:
            in_channles (int): Number of input channels for Atrous_Convolution.
            out_channles (int): Number of output channels for Atrous_Convolution.
        """
        super(ASSP, self).__init__()
        self.conv_1x1 = Atrous_Convolution(
            input_channels=in_channles, output_channels=out_channles,
            kernel_size=1, pad=0, dilation_rate=1)

        self.conv_6x6 = Atrous_Convolution(
            input_channels=in_channles, output_channels=out_channles,
            kernel_size=3, pad=6, dilation_rate=6)

        self.conv_12x12 = Atrous_Convolution(
            input_channels=in_channles, output_channels=out_channles,
            kernel_size=3, pad=12, dilation_rate=12)

        self.conv_18x18 = Atrous_Convolution(
            input_channels=in_channles, output_channels=out_channles,
            kernel_size=3, pad=18, dilation_rate=18)

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=in_channles, out_channels=out_channles,
                kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.final_conv = Atrous_Convolution(
            input_channels=out_channles * 5, output_channels=out_channles,
            kernel_size=1, pad=0, dilation_rate=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the ASSP module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the ASSP layers.
        """
        x_1x1 = self.conv_1x1(x)
        x_6x6 = self.conv_6x6(x)
        x_12x12 = self.conv_12x12(x)
        x_18x18 = self.conv_18x18(x)
        img_pool_opt = self.image_pool(x)
        img_pool_opt = F.interpolate(
            img_pool_opt, size=x_18x18.size()[2:],
            mode='bilinear', align_corners=True)
        # Concatenation of all features
        concat = torch.cat(
            (x_1x1, x_6x6, x_12x12, x_18x18, img_pool_opt),
            dim=1)
        x_final_conv = self.final_conv(concat)
        return x_final_conv


class ResNet_50(nn.Module):
    """
    A custom ResNet-50 model that allows specifying an output layer.

    Attributes:
        pretrained (nn.Module): The pretrained ResNet-50 model.
        output_layer (str): The name of the layer to use as the output layer.
        net (nn.Sequential): The modified ResNet-50 model up to the specified output layer.
    """
    def __init__(self, output_layer: str = None) -> None:
        """
        Initializes the ResNet_50 model.

        Args:
            output_layer (str, optional): The name of the layer to use as the output layer. 
                                          If None, the entire ResNet-50 model is used.
        """
        super(ResNet_50, self).__init__()
        self.pretrained = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.output_layer = output_layer

        if self.output_layer:
            layers = list(self.pretrained.children())
            layer_names = list(self.pretrained._modules.keys())
            output_index = layer_names.index(self.output_layer) + 1
            self.net = nn.Sequential(*layers[:output_index])
        else:
            self.net = nn.Sequential(*list(self.pretrained.children()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the network.
        """
        x = self.net(x)
        return x


class Deeplabv3Plus(nn.Module):
    """
    DeepLabv3+ model for semantic segmentation.

    Attributes:
        backbone (ResNet_50): The backbone ResNet-50 model up to 'layer3'.
        low_level_features (ResNet_50): The ResNet-50 model up to 'layer1' for low-level features.
        assp (ASSP): The Atrous Spatial Pyramid Pooling module.
        conv1x1 (Atrous_Convolution): 1x1 convolution layer for low-level features.
        conv_3x3 (nn.Sequential): 3x3 convolution layer for concatenated features.
        classifer (nn.Conv2d): Final classification layer.
    """

    def __init__(self, num_classes: int) -> None:
        """
        Initializes the Deeplabv3Plus model.

        Args:
            num_classes (int): Number of output classes for the segmentation task.
        """
        super(Deeplabv3Plus, self).__init__()

        self.backbone = ResNet_50(output_layer='layer3')
        self.low_level_features = ResNet_50(output_layer='layer1')
        self.assp = ASSP(in_channles=1024, out_channles=256)

        self.conv1x1 = Atrous_Convolution(
            input_channels=256, output_channels=48, kernel_size=1,
            dilation_rate=1, pad=0)

        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.classifer = nn.Conv2d(256, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the Deeplabv3Plus model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the network.
        """
        x_backbone = self.backbone(x)
        x_low_level = self.low_level_features(x)
        x_assp = self.assp(x_backbone)
        x_assp_upsampled = F.interpolate(
            x_assp, scale_factor=(4, 4),
            mode='bilinear', align_corners=True)
        x_conv1x1 = self.conv1x1(x_low_level)
        x_cat = torch.cat([x_conv1x1, x_assp_upsampled], dim=1)
        x_3x3 = self.conv_3x3(x_cat)
        x_3x3_upscaled = F.interpolate(
            x_3x3, scale_factor=(4, 4),
            mode='bilinear', align_corners=True)
        x_out = self.classifer(x_3x3_upscaled)
        return x_out
