import torch
import torch.optim
from torch import device
from typing import Tuple

from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference

from dataset import get_dataloaders
from model import Brats_model, get_loss_function


def train(data_dir: str, batch_size: int = 1, epochs: int = 50) -> None:
    """
    Main training loop for the 3D U-Net on the BraTS dataset.
    """

    # Device configuration
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training initialized on: {device}")

    # Data & Model
    train_loader, val_loader = get_dataloaders(data_dir, batch_size)
    model: Brats_model = Brats_model().to(device)

    # Loss & Optimizer
    loss_function = get_loss_function()
    optimizer: torch.optim.AdamW = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Metrics
    best_metric: float = -1.0
    dice_metric: DiceMetric = DiceMetric(include_background=False, reduction="mean")
    post_pred: AsDiscrete = AsDiscrete(threshold=0.75)

    for epoch in range(epochs):
        model.train()
        epoch_loss: float = 0.0

        for batch_data in train_loader:
            image, label = batch_data["image"].to(device), batch_data["label"].to(device)

            # Handle potential dimension issues
            if label.dim() == 4:
                label = label.unsqueeze(1)
            label = (label > 0).float()

            optimizer.zero_grad()
            output = model(image)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / len(train_loader):.4f}")

        # Validation phase
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                image_val, label_val = val_data["image"].to(device), val_data["label"].to(device)
                if label_val.dim() == 4:
                    label_val = label_val.unsqueeze(1)
                label_val = (label_val > 0).float()

                val_outputs = sliding_window_inference(
                    inputs=image_val, roi_size=(64, 64, 64), sw_batch_size=4,
                    overlap=0.5, predictor=model, mode="gaussian"
                )

                # Apply sigmoid and discrete transformation for evaluation
                val_outputs = [post_pred(i) for i in decollate_batch(torch.sigmoid(val_outputs))]
                label_val_list = decollate_batch(label_val)

                dice_metric(y_pred=val_outputs, y=label_val_list)

        metric: float = dice_metric.aggregate().item()
        dice_metric.reset()

        print(f"Epoch {epoch + 1}/{epochs} - Val Dice: {metric:.4f}")

        if metric > best_metric:
            best_metric = metric
            torch.save(model.state_dict(), "best_model.pth")
            print("New best model saved!")


if __name__ == "__main__":
    train(data_dir="/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation")