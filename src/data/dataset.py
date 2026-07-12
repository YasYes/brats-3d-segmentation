import os
import glob
from typing import List, Dict, Tuple
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, NormalizeIntensityd, RandSpatialCropd, EnsureTyped
)


def get_bts_data_lists(data_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Scans the BraTS 2020 directory and groups the 4 MRI modalities for each patient.
    Returns a tuple containing the training and validation data lists.
    """
    # Base paths
    base_train = os.path.join(data_dir, "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/")
    base_val = os.path.join(data_dir, "BraTS2020_ValidationData/MICCAI_BraTS2020_Validation_Data/")

    def scan_dir(path: str) -> List[Dict]:
        # Retrieve all patient directories matching the pattern
        patient_dirs = sorted(glob.glob(os.path.join(path, "BraTS20_*")))
        data_list = []
        for patient_dir in patient_dirs:
            # Search for the 4 distinct MRI modalities and the segmentation mask
            t1 = glob.glob(os.path.join(patient_dir, "*_t1.nii"))
            t1ce = glob.glob(os.path.join(patient_dir, "*_t1ce.nii"))
            t2 = glob.glob(os.path.join(patient_dir, "*_t2.nii"))
            flair = glob.glob(os.path.join(patient_dir, "*_flair.nii"))
            seg = glob.glob(os.path.join(patient_dir, "*_seg.nii"))

            # Ensure all files exist to form a valid training/validation pair
            if t1 and t1ce and t2 and flair and seg:
                data_list.append({
                    "image": [t1[0], t1ce[0], t2[0], flair[0]],
                    "label": seg[0]
                })
        return data_list

    return scan_dir(base_train), scan_dir(base_val)


def get_transforms(roi_size: Tuple[int, int, int] = (64, 64, 64)) -> Compose:
    """
    Builds the 3D medical image processing pipeline.
    Handles spatial geometric alignment and intensity normalization.
    """
    return Compose([
        # Load NIfTI files
        LoadImaged(keys=["image", "label"]),
        # Ensure channel-first format (C, D, H, W)
        EnsureChannelFirstd(keys=["image", "label"]),
        # Standardize anatomical orientation to RAS
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # Resample voxels to 1x1x1 mm using appropriate interpolation
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest")
        ),
        # Z-Score intensity normalization applied to images
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        # Random 3D cropping for GPU VRAM management
        RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False),
        # Convert to PyTorch tensors
        EnsureTyped(keys=["image", "label"])
    ])


def get_dataloaders(
        data_dir: str,
        batch_size: int = 1,
        roi_size: Tuple[int, int, int] = (64, 64, 64)
) -> Tuple[DataLoader, DataLoader]:
    """
    Initializes and returns the training and validation data loaders.
    """
    data_list, data_list_val = get_bts_data_lists(data_dir)
    transforms = get_transforms(roi_size)

    # Initialize datasets
    dataset = Dataset(data_list, transform=transforms)
    dataset_val = Dataset(data_list_val, transform=transforms)

    # Training DataLoader: shuffled for stochastic optimization
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    # Validation DataLoader: fixed order for consistent performance metrics
    dataloader_val = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return dataloader, dataloader_val