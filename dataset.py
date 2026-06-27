import glob
import os

from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, NormalizeIntensityd, \
    RandSpatialCropd, EnsureTyped

import os
import glob


def get_bts_data_list(data_dir: str):
    """
    Scans for patient folders regardless of the nested depth.
    """
    # On cherche récursivement tous les dossiers contenant 'BraTS20_Training_'
    search_pattern = os.path.join(data_dir, "**", "BraTS20_Training_*")
    patient_dirs = sorted(glob.glob(search_pattern, recursive=True))

    data_list = []

    for patient_dir in patient_dirs:
        # On vérifie si c'est bien un dossier (et non un fichier)
        if os.path.isdir(patient_dir):
            image_files = glob.glob(os.path.join(patient_dir, "*_t1ce.nii"))
            label_files = glob.glob(os.path.join(patient_dir, "*_seg.nii"))

            if image_files and label_files:
                data_list.append({
                    "image": image_files[0],
                    "label": label_files[0]
                })

    print(f"DEBUG: Scanned {len(patient_dirs)} potential folders. Found {len(data_list)} valid pairs.")
    return data_list

def get_transforms(roi_size=(64,64,64)):
    """
    Builds the 3D medical image processing pipeline.
    Handles both spatial (geometry) and intensity (radiometry) augmentations safely.
    """
    return Compose([
        # Load NIfTI files
        LoadImaged(keys=["image", "label"]),

        # Add channel dimension (C, H, W, D)
        EnsureChannelFirstd(keys=["image", "label"]),

        # Standardize anatomical orientation (Right, Anterior, Superior)
        Orientationd(keys=["image", "label"], axcodes="RAS"),

        # Standardize voxel size to 1x1x1 mm.
        # 'bilinear' for continuous MRI, 'nearest' to preserve discrete label classes (0,1,2,3).
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest")
        ),

        # Z-Score Normalization.
        # Applied only to the image. Altering the mask would destroy the class labels.
        NormalizeIntensityd(
            keys=["image"],
            nonzero=True,
            channel_wise=True
        ),

        # Random 3D cropping to manage GPU VRAM
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=roi_size,
            random_size=False
        ),

        # Convert numpy arrays to PyTorch Tensors
        EnsureTyped(keys=["image", "label"])
    ])

def get_dataloader(data_dir:str,batch_size:int=1,roi_size=(64,64,64)):

    data_list=get_bts_data_list(data_dir)

    transforms=get_transforms(roi_size)

    dataset=Dataset(data_list,transform=transforms)

    dataloader=DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True

    )
    return dataloader
