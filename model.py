import torch as nn
from monai.networks.nets.unet import UNet
from monai.losses import DiceLoss

class Brats_model(nn.Module):
    """
    3D U-Net Architecture optimized for the BraTS medical dataset.
    Handles multi-modal inputs and nested multi-label segmentation.
    """
    def __init__(self):
        super(Brats_model, self).__init__()

        self.net=UNet(
            spatial_dims=3,  # Specifies that inputs are 3D volumes (MRI), not 2D images
            in_channels=4,  # Stack of 4 MRI sequences: T1, T1Gd, T2, FLAIR
            out_channels=3,  # 3 overlapping tumor sub-regions: WT, TC, ET
            channels=(32, 64, 128, 256),  # Feature map filters (powers of 2 for hardware optimization)
            strides=(2, 2, 2),  # Downsampling steps for spatial reduction
            norm="GROUP"  # Group Normalization: prevents math crash when batch_size=1 due to VRAM limits
        )

    def forward(self,x):
        """
         Forward pass.
         Note: The network outputs raw logits, not probabilities.
        """
        return self.net(x)

def get_loss_function():
    """
    Returns the Dice Loss configured for independent marginal probabilities.
    Strictly avoids Softmax to prevent mathematical coupling between overlapping classes.
    """
    return DiceLoss(
        include_background=False,  # Exclude the black background (healthy tissue/air) from the loss calculation
        sigmoid=True,  # Applies Sigmoid internally to convert raw logits into independent probabilities
        squared_pred=True  # Squares the predictions in the denominator to stabilize gradients for small tumors
    )