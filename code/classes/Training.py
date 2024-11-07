import os
import sys
import random
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset
import torchvision.transforms.functional as Fv
import torchmetrics
from torchmetrics import JaccardIndex
from transformers import get_scheduler
from PIL import Image  # Importer Pillow pour charger les images TIF
import torchmetrics
from torchmetrics import JaccardIndex

# Define the project directory path
project_dir = '/content/gdrive/MyDrive/'

# Define the runs directory path within the project directory
runs_directory = os.path.join(project_dir, 'runs')

# Define the data directory path within the project directory
data_directory = os.path.join(project_dir, 'data')


class UnetSegmentor(nn.Module):
    def __init__(self, n_blocks=4, n_filter=32, num_classes=1, p=0.5):
        """
               Initializes the U-Net segmentation model.

               Args:
                   n_blocks (int): Number of blocks in the U-Net architecture. Default is 4.
                   n_filter (int): Number of filters in the first convolutional layer. Default is 32.
                   num_classes (int): Number of output classes for segmentation. Default is 1.
                   p (float): Dropout probability. Default is 0.5.
        """
        super(UnetSegmentor, self).__init__()
        self.n_blocks = n_blocks
        self.p = p
        self.input_conv = nn.Conv2d(in_channels=3, out_channels=n_filter, kernel_size=3, padding=1)
        self.encoder_convs = nn.ModuleList([
            self._create_encoder_conv_block(channels=n_filter * 2 ** i, kernel_size=3)
            for i in range(0, n_blocks - 1)
        ])
        self.mid_conv = self._create_encoder_conv_block(
            channels=n_filter * 2 ** (n_blocks - 1),
            kernel_size=3
        )
        self.decoder_deconvs = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=n_filter * 2 ** (i + 1),
                out_channels=n_filter * 2 ** i,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            )
            for i in reversed(range(n_blocks))
        ])
        self.decoder_convs = nn.ModuleList([
            self._create_decoder_conv_block(
                channels=n_filter * 2 ** i,
                kernel_size=3
            )
            for i in reversed(range(n_blocks))
        ])
        self.seg_conv = nn.Conv2d(
            in_channels=n_filter,
            out_channels=num_classes,
            kernel_size=3,
            padding=1
        )

    def _create_encoder_conv_block(self, channels, kernel_size):
        """
                Creates a convolutional block for the encoder part of the U-Net.

                Args:
                    channels (int): Number of input channels for the convolutional block.
                    kernel_size (int): Size of the convolutional kernel.

                Returns:
                    nn.Sequential: A sequential block containing convolutional layers, batch normalization, and ReLU activation.
        """
        return nn.Sequential(
            nn.Conv2d(channels, channels*2, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(channels*2),
            nn.ReLU(),
            nn.Conv2d(channels*2, channels*2, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(channels*2),
            nn.ReLU(),
        )

    def _create_decoder_conv_block(self, channels, kernel_size):
        """
              Creates a convolutional block for the decoder part of the U-Net.

              Args:
                  channels (int): Number of input channels for the convolutional block.
                  kernel_size (int): Size of the convolutional kernel.

              Returns:
                  nn.Sequential: A sequential block containing convolutional layers, batch normalization, and ReLU activation.
        """
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def forward(self, x):
        """
                Defines the forward pass of the U-Net model.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, num_classes, height, width) representing
                    the segmentation map.
        """
        feature_list = []
        x = self.input_conv(x)
        feature_list.append(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.dropout(x, p=self.p)
        for i in range(self.n_blocks-1):
            x = self.encoder_convs[i](x)
            feature_list.append(x)
            x = F.max_pool2d(x, kernel_size=2)
            x = F.dropout(x, p=self.p)

        x = self.mid_conv(x)

        for i in range(self.n_blocks):
            x = self.decoder_deconvs[i](x)
            x = F.dropout(x, p=self.p)
            x = self.decoder_convs[i](x + feature_list[::-1][i])

        return self.seg_conv(x)


class EfficientSegmentationDataset(VisionDataset):
    def __init__(self, data_dir, rock_names, num_classes=3, num_samples=None, crop_size=(224, 224), p=0.5, img_res=224):
        """
                Initializes the dataset for efficient segmentation.

                Args:
                    data_dir (str): Directory containing the image and mask files.
                    rock_names (list): List of rock names corresponding to the images and masks.
                    num_classes (int): Number of classes in the segmentation masks. Default is 3.
                    num_samples (int, optional): Number of samples to use. If None, it will be determined automatically.
                    crop_size (tuple): Size to which images will be cropped. Default is (224, 224).
                    p (float): Probability of applying augmentations. Default is 0.5.
                    img_res (int): Resolution to which images will be resized. Default is 224.
        """
        super().__init__()
        print("Loading data ...")
        self.img_data = []
        self.mask_data = []
        self.rock_names = rock_names
        self.crop_size = crop_size
        self.p = p
        self.IMG_RES = img_res
        self.num_classes = num_classes
        self.inference_mode = False

        for rock in rock_names:
            img_path = os.path.join(data_dir, f"{rock}_img.tif")
            mask_path = os.path.join(data_dir, f"{rock}_mask.tif")
            img = Image.open(img_path).convert("RGB")  # Load the TIF image
            mask = Image.open(mask_path).convert("L")  # Load the TIF mask in grayscale
            self.img_data.append(np.array(img))
            self.mask_data.append(np.array(mask))

        if num_samples is None:
            self.num_samples = len(self.img_data[0])
        else:
            self.num_samples = num_samples

        self.num_datasets = len(self.img_data)

    def __getitem__(self, idx):
        """
               Retrieves an image and its corresponding mask from the dataset.

               Args:
                   idx (int): Index of the sample to retrieve.

               Returns:
                   tuple: A tuple containing the image tensor and the mask tensor.
        """
        try:
            dataset_index = idx % self.num_datasets
            data_idx = (idx // self.num_datasets)

            img = self.img_data[dataset_index][data_idx]
            mask = self.mask_data[dataset_index][data_idx]

            if img is None or mask is None:
                raise ValueError("Image or mask is None.")

            # Convert to tensors
            img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous() / 255.0
            mask = torch.from_numpy(mask).contiguous() / 255.0

            img = F.interpolate(
                input=img.unsqueeze(0),
                size=(self.IMG_RES, self.IMG_RES),
                mode="bicubic",
                align_corners=False
            ).squeeze()

            mask = F.interpolate(
                input=mask.unsqueeze(0).unsqueeze(0),
                size=(self.IMG_RES, self.IMG_RES),
                mode="nearest"
            ).squeeze()

            # Apply augmentations if necessary
            if torch.rand(1) < self.p and not self.inference_mode:
                img = torchvision.transforms.functional.hflip(img)
                mask = torchvision.transforms.functional.hflip(mask)

            if torch.rand(1) < self.p and not self.inference_mode:
                img = torchvision.transforms.functional.vflip(img)
                mask = torchvision.transforms.functional.vflip(mask)

            return img, mask

        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            return torch.zeros(3, self.IMG_RES, self.IMG_RES), torch.zeros(self.IMG_RES, self.IMG_RES)

    def __len__(self):
        """
               Returns the total number of samples in the dataset.

               Returns:
                   int: Total number of samples.
        """
        return self.num_datasets * self.num_samples


def load_segmentation_data(data_dir, train_rocks, test_rocks, num_classes=1, val_split=0.8, batch_size=2):
    """
       Loads the segmentation data for training and validation.

       Args:
           data_dir (str): Directory containing the image and mask files.
           train_rocks (list): List of rock names for the training dataset.
           test_rocks (list): List of rock names for the testing dataset.
           num_classes (int): Number of classes in the segmentation masks. Default is 1.
           val_split (float): Proportion of the dataset to use for validation. Default is 0.8.
           batch_size (int): Number of samples per batch. Default is 2.

       Returns:
           dict: A dictionary containing the training and validation DataLoaders and the number of classes.
    """
    # Initialize the training dataset with specified parameters
    train_dataset = EfficientSegmentationDataset(
        data_dir, train_rocks, num_classes=num_classes
    )

    # Initialize the testing dataset with specified parameters
    test_dataset = EfficientSegmentationDataset(
        data_dir, test_rocks, num_classes=num_classes
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return {'train': train_loader, 'val': val_loader, 'num_classes': num_classes}
