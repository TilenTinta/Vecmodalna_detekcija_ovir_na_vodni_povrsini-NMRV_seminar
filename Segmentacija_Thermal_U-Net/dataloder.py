from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import functional as F
from albumentations.pytorch import ToTensorV2
from albumentations import (
    HorizontalFlip, RandomRotate90, ShiftScaleRotate, Resize, Normalize,
    GaussianBlur, RandomBrightnessContrast, CLAHE, HueSaturationValue,
    CoarseDropout, ElasticTransform, GridDistortion, Compose, Affine
)
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import random_split


class LadicaDataset(VisionDataset):
    def __init__(self, root, file_list, transform_image, transform_mask):
        super(LadicaDataset, self).__init__(root)
        self.root = Path(root)
        self.image_folder = self.root / "thermal_images"
        self.mask_folder = self.root / "thermal_semantic_annotations"

        # PReberi imena iz txt fila
        with open(file_list, 'r') as f:
            self.image_files = [line.strip() + ".png" for line in f.readlines()]

        self.transform_image = transform_image
        self.transform_mask = transform_mask

    def __getitem__(self, index):
        image_path = self.image_folder / self.image_files[index]
        mask_path = self.mask_folder / self.image_files[index]

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # prebere slike
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) # prebere maske - sivinske

        if image is None or mask is None:
            return self.__getitem__((index + 1) % len(self))  # Preskoči in vzemi naslednjo

        if image.dtype == np.uint16:  # If 16-bit thermal image
            image = cv2.normalize(image, None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)  
            image = cv2.convertScaleAbs(image, alpha=255/(2**16))  # Convert to 8-bit

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Ensure it's RGB

        # if len(image.shape) == 2:  # Enokanalna slika (16-bitna)
        #     #image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #     image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Pretvori v RGB
        # elif len(image.shape) == 3 and image.shape[2] == 3:  # Slika je že RGB
        #     pass  # Ne narest nič
        # else:
        #     raise ValueError(f"Unexpected image shape: {image.shape} for {image_path}")

        # Augmentacija slik in mask (oboje isto)
        if self.transform_image:
            augmented = self.transform_image(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Convert to PyTorch tensors
        image = image.clone().detach() #/ 255.0  # Normalizacija v [0, 1]

        # Omeji na vrednosti mask [0, 4], 255 je za pytorch ignore
        mask = np.where(mask == 5, 3, np.where(mask > 4, 255, mask)) # nad 4 jse vse 255 razen 5 je razred 3 
        mask = np.where(mask != 255, mask - 1, 255)  # [1,2,3,4] → [0,1,2,3]

        # Convert to tensor
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long().clone().detach()
        else:
            mask = mask.long().clone().detach()

        return image, mask, self.image_files[index]

    # Vrne število slik iz map
    def __len__(self):
        return len(self.image_files)
    

# augmentacija
def get_transforms(size):

    transform_image = Compose([
        HorizontalFlip(p=0.5), # p = verjetnost (50% da se bo funkcija izvedla)
        #RandomRotate90(p=0.5),
        Affine(shear={"x": (-10, 10)}, p=0.3),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        GaussianBlur(blur_limit=(3, 7), p=0.3),  # Zameglitev
        #RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),  # Spreminjanje svetlosti in kontrasta
        #CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.2),  # Izboljšanje kontrasta
        #HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),  # Barvne spremembe
        Resize(width=size[0], height=size[1]),
        #Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # Dodano
        ToTensorV2()
    ], is_check_shapes=False)
    
    transform_mask = Compose([  
        HorizontalFlip(p=0.5),
        #RandomRotate90(p=0.5),
        Affine(shear={"x": (-10, 10)}, p=0.3),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        Resize(width=size[0], height=size[1]),
        ToTensorV2()
    ], is_check_shapes=False)
    
    return transform_image, transform_mask


# dataloader za train in test
def get_dataloaders(data_dir, batch_size, size):

    transform_image, transform_mask = get_transforms(size)

    # Load the full training dataset
    full_train_dataset = LadicaDataset(
        data_dir,
        file_list=os.path.join(data_dir, "train.txt"),
        transform_image=transform_image,
        transform_mask=transform_mask
    )

    # Split dataset into training (2/3) and validation (1/3)
    train_size = int(2 * len(full_train_dataset) / 3)
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Load the test dataset (remains unchanged)
    test_dataset = LadicaDataset(
        data_dir,
        file_list=os.path.join(data_dir, "test.txt"),
        transform_image=Compose([
            Resize(width=size[0], height=size[1]),
            ToTensorV2()
        ], is_check_shapes=False),
        transform_mask=Compose([ 
            Resize(width=size[0], height=size[1]), 
            ToTensorV2()
        ], is_check_shapes=False)
    )

    # Data loaders: Train, Validtion, Test
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8,  
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        batch_size=batch_size, 
        shuffle=False,  
        num_workers=8,  
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=8, 
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
