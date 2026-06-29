from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, NormalizeIntensityd, \
    RandSpatialCropd, EnsureTyped
import os
import glob


def get_dataloaders(data_dir: str):
    """
    Scans the BraTS 2020 directory and groups the 4 MRI modalities for each patient.
    Returns a list of dictionaries containing paths to the stacked images and their labels.
    """
    # Define the absolute base path validated on Kaggle
    base = "/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"

    # Retrieve all patient directories matching the expected pattern
    patient_dirs = sorted(glob.glob(os.path.join(base, "BraTS20_Training_*")))

    data_list = []

    for patient_dir in patient_dirs:
        # Search for the 4 distinct MRI modalities
        t1 = glob.glob(os.path.join(patient_dir, "*_t1.nii"))
        t1ce = glob.glob(os.path.join(patient_dir, "*_t1ce.nii"))
        t2 = glob.glob(os.path.join(patient_dir, "*_t2.nii"))
        flair = glob.glob(os.path.join(patient_dir, "*_flair.nii"))

        # Search for the corresponding segmentation mask
        seg = glob.glob(os.path.join(patient_dir, "*_seg.nii"))

        # Ensure all 5 files (4 modalities + 1 mask) exist before creating a pair
        if t1 and t1ce and t2 and flair and seg:
            data_list.append({
                # MONAI will automatically stack this list into a 4-channel 3D tensor during data loading
                "image": [t1[0], t1ce[0], t2[0], flair[0]],
                "label": seg[0]
            })

        # Define the absolute base path validated on Kaggle
        base_val = "/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/"

        # Retrieve all patient directories matching the expected pattern
        patient_dirs_val = sorted(glob.glob(os.path.join(base_val, "BraTS20_Validation_*")))

        data_list_val = []

        for patient_dir in patient_dirs_val:
            # Search for the 4 distinct MRI modalities
            t1 = glob.glob(os.path.join(patient_dir, "*_t1.nii"))
            t1ce = glob.glob(os.path.join(patient_dir, "*_t1ce.nii"))
            t2 = glob.glob(os.path.join(patient_dir, "*_t2.nii"))
            flair = glob.glob(os.path.join(patient_dir, "*_flair.nii"))

            # Search for the corresponding segmentation mask
            seg = glob.glob(os.path.join(patient_dir, "*_seg.nii"))

            # Ensure all 5 files (4 modalities + 1 mask) exist before creating a pair
            if t1 and t1ce and t2 and flair and seg:
                data_list_val.append({
                    # MONAI will automatically stack this list into a 4-channel 3D tensor during data loading
                    "image": [t1[0], t1ce[0], t2[0], flair[0]],
                    "label": seg[0]
                })
    return data_list, data_list_val

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
