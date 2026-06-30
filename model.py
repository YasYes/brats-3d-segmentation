import torch
import torch.nn as nn
from monai.networks.nets.unet import UNet
from monai.losses import DiceLoss
from typing import Tuple

class Brats_model(nn.Module):
    """
    3D U-Net Architecture optimized for the BraTS medical dataset.
    Handles multi-modal inputs and nested multi-label segmentation.
    """
    def __init__(self) -> None:
        super(Brats_model, self).__init__()

        self.net: nn.Module = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2),
            norm=("GROUP", {"num_groups": 2})
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns: Raw logits.
        """
        return self.net(x)

def get_loss_function() -> DiceLoss:
    """
    Returns the Dice Loss configured for independent marginal probabilities.
    """
    return DiceLoss(
        to_onehot_y=True,
        include_background=False,
        sigmoid=True,
        squared_pred=True
    )