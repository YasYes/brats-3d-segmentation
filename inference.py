import torch
import nibabel as nib
import numpy as np
from typing import Tuple, List
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, NormalizeIntensityd, \
    EnsureTyped

from model import Brats_model


def run_inference(
        model_path: str,
        image_paths: List[str],
        label_paths: List[str],
        output_path: str
) -> Tuple[np.ndarray, np.ndarray, float]:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: Brats_model = Brats_model().to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Pipeline de transformation (doit être identique à l'entraînement !)
    loader: Compose = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"])
    ])

    loaded_data = loader({"image": image_paths, "label": label_paths})

    # Ajout de la dimension batch (B, C, D, H, W)
    image_tensor: torch.Tensor = loaded_data["image"].unsqueeze(0).to(device)
    label_tensor: torch.Tensor = loaded_data["label"].unsqueeze(0).to(device)

    # Standardisation du label
    if label_tensor.dim() == 4:
        label_tensor = label_tensor.unsqueeze(1)
    label_tensor = (label_tensor > 0).float()

    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=image_tensor,
            roi_size=(64, 64, 64),
            sw_batch_size=4,
            predictor=model,
            overlap=0.5,
            mode="gaussian"
        )

    probabilities: torch.Tensor = torch.sigmoid(logits)
    prediction: torch.Tensor = (probabilities > 0.75).float()

    dice_metric(y_pred=prediction, y=label_tensor)
    final_dice: float = dice_metric.aggregate().item()
    print(f"Inference complete! 3D Dice Score: {final_dice:.4f}")

    # Sauvegarde NIfTI
    pred_numpy: np.ndarray = prediction.squeeze(0).cpu().numpy().astype(np.uint8)
    # Si ton modèle prédit plusieurs classes, tu peux choisir un canal spécifique ou enregistrer le tout
    nifti_image = nib.Nifti1Image(pred_numpy.transpose(1, 2, 3, 0), loaded_data["image"].meta["affine"])
    nib.save(nifti_image, output_path)

    return loaded_data["image"][1].cpu().numpy(), pred_numpy, final_dice