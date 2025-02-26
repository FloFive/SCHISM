import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms.functional as func_torch
from torchvision.datasets import VisionDataset
import torch.nn.functional as nn_func
from patchify import patchify

class TiffDatasetLoader(VisionDataset):

    def __init__(self, img_data=None, mask_data=None, indices=None, data_stats=None,
                 num_classes=None, img_res=560, crop_size=(224, 224), p=0.5, inference_mode=False, ignore_background=True):
        """
        Initializes the TiffDatasetLoader with image and mask data.

        Args:
            img_data (dict): A dictionary containing image file paths.
            mask_data (dict): A dictionary containing mask file paths.
            indices (list): A list of tuples indicating the dataset and sample indices.
            data_stats (dict): A dictionary containing normalization statistics.
            num_classes (int): The number of classes in the dataset.
            img_res (int): The resolution to which images will be resized.
            crop_size (tuple): The size of the crop to be applied to images.
            p (float): Probability of applying random transformations (flips).
            inference_mode (bool): Flag indicating if the dataset is in inference mode.
        """
        super().__init__(transforms=None)
        self.data_stats = data_stats
        self.img_data = img_data
        self.mask_data = mask_data
        self.indices = indices
        self.num_classes = num_classes
        self.crop_size = crop_size
        self.img_res = img_res
        self.inference_mode = inference_mode
        self.p = p
        self.ignore_background = ignore_background
        self.image_dims = self.get_image_dimensions()
        if not self.inference_mode:
            self.class_values = self._compute_class_values()

    def get_image_dimensions(self):
        """
        Reads the dimensions of the first image from the dataset dynamically.
        Assumes all images have the same dimensions.

        Returns:
            tuple: A tuple containing the height and width of the image.
        """
        dataset_id, sample_id = self.indices[0]
        img_path = self.img_data[dataset_id][sample_id]
        with Image.open(img_path) as img:
            return img.size[::-1]

    def get_random_crop_params(self):
        """
        Generates random cropping parameters for the images.

        Returns:
            tuple: A tuple containing the starting row index, starting column index, height, and width of the crop.

        Raises:
            ValueError: If the required crop size is larger than the input image size.
        """
        h, w = self.image_dims
        th, tw = self.crop_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()

        return i, j, th, tw
        
    def get_valid_crop(self, img, mask, threshold=0.8, max_attempts=10):
        """
        Attempts to find a crop of the image and mask where the fraction of background pixels
        (with value 0 in the mask) is below the specified threshold. Falls back to center crop if no valid crop is found.
    
        Args:
            img (np.array): The input image (H x W x C).
            mask (np.array): The segmentation mask (H x W), where 0 indicates background.
            threshold (float): Maximum allowed fraction of background pixels (default 0.8).
            max_attempts (int): Maximum number of crop attempts before falling back to center crop.
    
        Returns:
            tuple: Cropped image and mask.
        """
        for attempt in range(max_attempts):
            # Get random crop parameters
            i, j, h, w = self.get_random_crop_params()
            # Crop the mask
            crop_mask = mask[i:i+h, j:j+w].copy()
            # Calculate the fraction of background pixels
            background_ratio = (crop_mask == 0).sum() / crop_mask.size
    
            if background_ratio < threshold:
                # Valid crop found, return it
                crop_img = img[i:i+h, j:j+w, :].copy()
                return crop_img, crop_mask
    
        # If max_attempts reached, perform center crop
        h, w = self.image_dims
        th, tw = self.crop_size
        center_i = (h - th) // 2
        center_j = (w - tw) // 2
    
        crop_img = img[center_i:center_i+th, center_j:center_j+tw, :].copy()
        crop_mask = mask[center_i:center_i+th, center_j:center_j+tw].copy()
    
        return crop_img, crop_mask

    def extract_patches(self, img_np):
        height, width = self.image_dims
        patch_h, patch_w = self.crop_size 

        # Transpose to [H, W, C] for padding
        img_np = np.transpose(img_np, (1, 2, 0))

        # Check if padding is necessary (non-square or not a multiple of crop size)
        if height != width or height % patch_h != 0 or width % patch_w != 0:
            # Calculate the padding needed to reach the nearest multiple of crop size (224)
            pad_height = (patch_h - height % patch_h) % patch_h
            pad_width = (patch_w - width % patch_w) % patch_w

            # Correct padding tuple for a 3D image (H, W, C)
            padding = [(0, pad_height), (0, pad_width), (0, 0)]  # Pad along height and width, no padding for channels

            # Pad the image
            img_np = np.pad(img_np, padding, mode='constant', constant_values=0)

        # Transpose back to [C, H, W] after padding
        img_np = np.transpose(img_np, (2, 0, 1)).squeeze()
        patches = patchify(img_np, (img_np.shape[0], patch_h, patch_h), step=patch_h)
        patches = patches.reshape(-1, img_np.shape[0], patch_h, patch_w)  # [num_patches, C, patch_h, patch_w]
        return patches
    
    def _compute_class_values(self):
        """
        Computes unique class values from all available masks.

        Returns:
            list: Sorted list of unique class values in the dataset.
        """
        unique_values = set()
        for dataset_id, sample_id in self.indices[:10]:  # Checking a subset for efficiency
            mask_path = self.mask_data[dataset_id][sample_id]
            mask = np.array(Image.open(mask_path).convert("L"))
            unique_values.update(np.unique(mask))
            sorted_values = sorted(unique_values)

        if self.ignore_background and len(sorted_values) > 1:
            sorted_values.pop(0)  # Remove the first value assuming it's the background
        
        return sorted_values
    
    def _weights_calc(self, mask, temperature=50.0):
        """
        Computes class weights based on mask pixel frequencies.

        Args:
            mask (np.array): The segmentation mask.
            temperature (float): Temperature parameter for softmax scaling.

        Returns:
            torch.Tensor: Computed class weights.
        """
        counts = np.bincount(mask.astype(int).ravel(), minlength=max(self.class_values) + 1)
        counts = counts[self.class_values]  # Select only relevant class counts

        class_ratio = counts / np.sum(counts)
        u_weights = 1 / class_ratio
        weights = np.nan_to_num(u_weights, posinf=-np.inf)
        weights = nn_func.softmax(torch.from_numpy(weights).float() / temperature, dim=-1)
        
        if torch.any(torch.isnan(weights)):
            print("NaN encountered in weights:", weights)
            print("Class ratios:", class_ratio)
            print("Unnormalized weights:", u_weights)
            raise ValueError("Invalid weights calculation")
        
        return weights

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding mask by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the normalized image tensor, mask tensor (if not inference mode),
                   dataset ID, and image path.

        Raises:
            AssertionError: If the dimensions of the image and mask do not match.
        """
        dataset_id, sample_id = self.indices[idx]
        img_path = self.img_data[dataset_id][sample_id]
        
        if dataset_id in self.data_stats:
            m, s = self.data_stats[dataset_id]
        else:
            m, s = self.data_stats["default"]

        if self.inference_mode:
            img = np.array(Image.open(img_path).convert("RGB"))
            img_tensor = torch.from_numpy(img).float() 
            img_tensor = img_tensor.permute(2, 0, 1).contiguous() / 255.0
            img_normalized = torchvision.transforms.functional.normalize(img_tensor, mean=m, std=s)
            patches = self.extract_patches(img_normalized)
            processed_patches = []

            for i in range(patches.shape[0]):  # Loop through patch grid
                patch = patches[i]
                patch_tensor = torch.tensor(patch).unsqueeze(0)
                patch_resized = nn_func.interpolate(patch_tensor, size=(self.img_res, self.img_res),
                                                    mode="bicubic", align_corners=False).squeeze()  
                processed_patches.append(patch_resized)

            return processed_patches, dataset_id, img_path

        mask_path = self.mask_data[dataset_id][sample_id]
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))            

        assert img.shape[:2] == mask.shape, (
            f"Mismatch in dimensions: Image {img.shape} vs Mask {mask.shape} for {img_path}"
        )
        
        img, mask = self.get_valid_crop(img, mask, threshold=0.8, max_attempts=20)

        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).contiguous() / 255
        unique_vals = np.unique(mask)
        mask = (mask - unique_vals.min()) / (unique_vals.max() - unique_vals.min()) # Normalize all values between 0 and 1
        mask_tensor = torch.from_numpy(mask).contiguous()
        
        weights = self._weights_calc(np.array(mask_tensor))
        print(weights)
        img_resized = nn_func.interpolate(img_tensor.unsqueeze(0), size=(self.img_res, self.img_res),
                                          mode="bicubic", align_corners=False).squeeze()
        mask_resized = nn_func.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0), size=(self.img_res, self.img_res),
                                           mode="nearest").squeeze()
        if torch.rand(1).item() < self.p:
            img_resized = torchvision.transforms.functional.hflip(img_resized)
            mask_resized = torchvision.transforms.functional.hflip(mask_resized)

        if torch.rand(1).item() < self.p:
            img_resized = torchvision.transforms.functional.vflip(img_resized)
            mask_resized = torchvision.transforms.functional.vflip(mask_resized)

        #TODO: as many image treatment as required
        #shear
        
        img_normalized = torchvision.transforms.functional.normalize(img_resized, mean=m, std=s).float()
 
        if self.num_classes >= 2:
            if self.ignore_background:
                # ignore index is -1
                mask_resized = (mask_resized * self.num_classes).long() - 1
            else:
                # no ignore index but rescale to int
                mask_resized = (mask_resized * (self.num_classes- 1)).long()

        return img_normalized, mask_resized, weights

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.indices)
