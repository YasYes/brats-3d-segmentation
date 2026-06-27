import torch.optim
from torch import device

from dataset import get_dataloader
from model import Brats_model, get_loss_function


def train(data_dir:str, batch_size :int=1, epochs: int=50):
    """
    Main training loop for the 3D U-Net on the BraTS dataset.
    """

    # Hardware configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training initialized on: {device}")

    # Component initialization
    dataloader=get_dataloader(data_dir,batch_size)
    model = Brats_model().to(device)

    # Loss configured for independent marginal probabilities
    loss_function=get_loss_function()

    # AdamW optimizer handles unstable gradients better than SGD for medical imaging
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    for epoch in range (epochs):
        model.train()
        epoch_loss = 0.0

        # Main training loop
        for batch_data in dataloader:

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
        epoch_loss /= len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}")

        # Smart Checkpointing
        if epoch == 0:
            best_loss = epoch_loss

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved! Loss improved: {best_loss:.4f}")

if __name__ == "__main__":
    # Point this to your local or Kaggle dataset path
    train(data_dir="/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation")





