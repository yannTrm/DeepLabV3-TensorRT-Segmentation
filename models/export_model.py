""" 
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ File for exporting model (Here deeplabv3) to ONNX format                 │
  └──────────────────────────────────────────────────────────────────────────┘
"""

#* Import

import torch
import torchvision.models as models
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights

def export_deeplabv3_to_onnx(model_path):
    """
    Export the DeepLabV3 model to ONNX format.

    Parameters:
    model_path (str): The path to save the ONNX model.
    """
    model = models.segmentation.deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT) # up-to-date weights
    model.eval()  

    dummy_input = torch.randn(1, 3, 224, 224) # Image with 3 channels, 224x224 pixels

    torch.onnx.export(
        model, 
        dummy_input, 
        model_path, 
        opset_version=11,
        input_names=["input"], 
        output_names=["output"]
    )

    print(f"Model exported to {model_path} successfully!")


if __name__ == "__main__":
    onnx_model_path = "./deeplabv3.onnx"
    export_deeplabv3_to_onnx(onnx_model_path)
