from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, NormalizeIntensityd, \
    RandSpatialCropd, EnsureTyped

import os
import glob


def get_bts_data_list(data_dir: str):
    """
    Scans the BraTS 2020 directory structure using the validated absolute path.
    """
    # Using the absolute path validated by your diagnostic script
    base = "/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"

    # Searching for all patient directories matching the pattern
    search_path = os.path.join(base, "BraTS20_Training_*")
    patient_dirs = sorted(glob.glob(search_path))

    print(f"DEBUG: Je cherche dans {base}, dossiers trouvés : {len(patient_dirs)}")

    data_list = []

    for patient_dir in patient_dirs:
        # Looking for files ending exactly with _t1ce.nii and _seg.nii
        image_files = glob.glob(os.path.join(patient_dir, "*_t1ce.nii"))
        label_files = glob.glob(os.path.join(patient_dir, "*_seg.nii"))

        # We only add the pair if both files are present
        if image_files and label_files:
            data_list.append({
                "image": image_files[0],
                "label": label_files[0]
            })

    # Final check to confirm we found valid data
    print(f"DEBUG: Nombre de paires (image, label) créées : {len(data_list)}")

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
