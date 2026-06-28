import monai
import torch
import nibabel as nib
import numpy as np
from monai.inferers import sliding_window_inference

from model import Brats_model


def run_inference(model_path,image_paths,output_path):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=Brats_model().to(device)

    model.load_state_dict(torch.load(model_path,map_location=device))
    model.eval()

    data_dict={"image":image_paths}

    loader=monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=["image"]),
        monai.transforms.EnsureChannelFirstd(keys=["image"]),
        monai.transforms.Orientationd(keys=["image"], axcodes = "RAS"),
        monai.transforms.Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear")
        ),
        monai.transforms.NormalizeIntensityd(
            keys=["image"],
            nonzero=True,
            channel_wise=True
        ),
        monai.transforms.EnsureTyped(keys=["image"])
    ])

    loaded_data=loader(data_dict)

    image_tensor=loaded_data["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits=sliding_window_inference(
            inputs=image_tensor,
            roi_size=(64,64,64),
            sw_batch_size=4,
            predictor=model
        )

    probabilities=torch.sigmoid(logits)
    prediction=(probabilities>0.5).float()

    pred_numpy=prediction.squeeze(0).cpu().numpy()

    if pred_numpy.shape[0]==2:
        pred_numpy=pred_numpy[1]

    original_affine=loaded_data["image"].meta["affine"]
    nifti_image = nib.Nifti1Image(pred_numpy.astype(np.uint8), original_affine)
    nib.save(nifti_image,output_path)

    return loaded_data["image"][1].cpu().numpy(), pred_numpy

