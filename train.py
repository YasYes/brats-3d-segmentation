import torch
import torch.optim
from torch import device

from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference

from dataset import get_dataloaders
from model import Brats_model, get_loss_function


def train(data_dir:str, batch_size :int=1, epochs: int=50):
    """
    Main training loop for the 3D U-Net on the BraTS dataset.
    """

    # Hardware configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training initialized on: {device}")

    # Component initialization
    train_loader,val_loader=get_dataloaders(data_dir,batch_size)
    model = Brats_model().to(device)

    # Loss configured for independent marginal probabilities
    loss_function=get_loss_function()

    # AdamW optimizer handles unstable gradients better than SGD for medical imaging
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    best_metric=-1.0
    best_metric_epoch=-1

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(threshold=0.75)

    for epoch in range (epochs):
        model.train()
        epoch_loss = 0.0

        # Main training loop
        for batch_data in train_loader:

            # Move data to GPU VRAM
            image=batch_data["image"].to(device)
            label=batch_data["label"].to(device)

            if len(label.shape) == 4:
                label = label.unsqueeze(1)
            label = (label > 0).float()

            optimizer.zero_grad()
            output=model(image)
            loss=loss_function(output,label)
            loss.backward()
            optimizer.step()

            # Accumulate loss for monitoring
            epoch_loss += loss.item()

        # Minimalist progress tracking
        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}")

        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                image_val=val_data["image"].to(device)
                label_val=val_data["label"].to(device)

                if len(label_val.shape) == 4:
                    label_val=label_val.unsqueeze(1)
                label_val = (label_val > 0).float()

                val_outputs=sliding_window_inference(
                    inputs=image_val,
                    roi_size=(64,64,64),
                    sw_batch_size=4,
                    overlap=0.5,
                    predictor=model,
                    mode="gaussian"
                )

                val_outputs=torch.sigmoid(val_outputs)

                val_outputs_list=decollate_batch(val_outputs)
                label_val_list=decollate_batch(label_val)


                val_output_convert=[post_pred(pred) for pred in val_outputs_list]

                dice_metric(y_pred=val_output_convert,y=label_val_list)

        metric = dice_metric.aggregate().item()

        # On remet la calculatrice à zéro pour la prochaine époque
        dice_metric.reset()

        print(f"Epoch {epoch + 1}/{epochs} - Val Dice: {metric:.4f}")

        # Smart Checkpointing basé sur la validation
        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved! Dice improved to: {best_metric:.4f}")

if __name__ == "__main__":
    # Point this to your local or Kaggle dataset path
    train(data_dir="/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation")





