from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import functional as F
from albumentations import Compose, HorizontalFlip, RandomRotate90, ShiftScaleRotate, LongestMaxSize, PadIfNeeded
from albumentations.pytorch import ToTensorV2
from albumentations import Resize
from albumentations import Normalize
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
        self.image_folder = self.root / "RGB_images"
        self.mask_folder = self.root / "RGB_semantic_annotations"

        # PReberi imena iz txt fila
        with open(file_list, 'r') as f:
            self.image_files = [line.strip() + ".png" for line in f.readlines()]

        self.transform_image = transform_image
        self.transform_mask = transform_mask

    def __getitem__(self, index):
        image_path = self.image_folder / self.image_files[index]
        mask_path = self.mask_folder / self.image_files[index]

        image = cv2.imread(image_path) # prebere slike
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) # prebere maske - sivinske

        # error check
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        # # Show the image and mask after rotation
        # plt.figure(figsize=(10, 5))

        # # Display the rotated image
        # plt.subplot(1, 2, 1)
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct colors
        # plt.title("Rotated Image")
        # plt.axis("off")

        # # Display the rotated mask
        # plt.subplot(1, 2, 2)
        # plt.imshow(mask, cmap='gray')  # Mask is grayscale
        # plt.title("Rotated Mask")
        # plt.axis("off")

        # plt.show()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR v RGB 

        # Augmentacija slik in mask (oboje isto)
        if self.transform_image:
            augmented = self.transform_image(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Convert to PyTorch tensors
        image = image.clone().detach() / 255.0  # Normalizacija v [0, 1]

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
        RandomRotate90(p=0.5),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        Resize(width=size[0], height=size[1]),
        ToTensorV2()
    ], is_check_shapes=False)
    
    transform_mask = Compose([  
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5),
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
