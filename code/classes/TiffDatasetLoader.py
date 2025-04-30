import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as func_torch
from torchvision.datasets import VisionDataset
import torch.nn.functional as nn_func
from patchify import patchify
import tifffile
from pathlib import Path

class TiffDatasetLoader(VisionDataset):

    def __init__(self, img_data=None, mask_data=None, indices=None, data_stats=None,
                 num_classes=None, img_res=560, crop_size=(224, 224), p=0.5,
                 inference_mode=False, ignore_background=True, weights=False):
        """
        Initializes the TiffDatasetLoader with image and mask data.
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
        self.weights = weights
        self.image_dims = self.get_image_dimensions()
        if not self.inference_mode:
            self.class_values = self._compute_class_values()

    def get_image_dimensions(self):
        """
        Determines the dimensions of the first image in the dataset.
        Assumes all images have the same dimensions.
        
        Returns:
            tuple: (height, width) of the image.
        
        Raises:
            FileNotFoundError: If the image file does not exist.
            RuntimeError: If the image cannot be opened.
            ValueError: If the image shape is unexpected.
        """
        dataset_id, sample_id = self.indices[0]
        img_path = Path(self.img_data[dataset_id][sample_id]).resolve()

        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")

        try:
            img = tifffile.imread(str(img_path))
        except Exception as e:
            raise RuntimeError(f"Failed to open image with tifffile: {e}")

        if img.ndim == 2:  # Grayscale image
            return img.shape
        elif img.ndim == 3:  # Multi-channel image
            return img.shape[:2]
        else:
            raise ValueError(f"Unexpected image shape {img.shape} for {img_path}")

    def get_random_crop_params(self):
        """
        Generates random cropping parameters for the images.
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
        is below the specified threshold and at least one class pixel is present. Falls back to center crop if no valid crop is found.
        """
        for _ in range(max_attempts):
            i, j, h, w = self.get_random_crop_params()
            crop_mask = mask[i:i+h, j:j+w].copy()
            
            # Calculate the background ratio
            background_ratio = (crop_mask == 0).sum() / crop_mask.size
            
            # Check if there is at least one non-background pixel
            has_class_pixel = (crop_mask != 0).any()
            
            if background_ratio < threshold and has_class_pixel:
                crop_img = img[i:i+h, j:j+w, ...].copy()
                return crop_img, crop_mask

        # Fallback to center crop
        h, w = self.image_dims
        th, tw = self.crop_size
        center_i = (h - th) // 2
        center_j = (w - tw) // 2
        crop_img = img[center_i:center_i+th, center_j:center_j+tw, ...].copy()
        crop_mask = mask[center_i:center_i+th, center_j:center_j+tw].copy()
        return crop_img, crop_mask

    def extract_patches(self, img_np):
        """
        Splits the image into patches using patchify.
        """
        patch_h, patch_w = self.crop_size  # Expected patch size (e.g., (224, 224))
        _, actual_height, actual_width = img_np.shape

        # Pad if needed so that the image dimensions are multiples of patch_h and patch_w
        if actual_height % patch_h != 0 or actual_width % patch_w != 0:
            pad_height = (patch_h - actual_height % patch_h) % patch_h
            pad_width = (patch_w - actual_width % patch_w) % patch_w
            padding = [(0, 0), (0, pad_height), (0, pad_width)]
            img_np = np.pad(img_np, padding, mode='constant', constant_values=0)

        patches = patchify(img_np, (img_np.shape[0], patch_h, patch_w), step=patch_h)
        return patches.reshape(-1, img_np.shape[0], patch_h, patch_w)

    def _compute_class_values(self):
            """
            Computes unique class values from all available masks.
            
            Returns:
                list: Sorted list of unique class values in the dataset.
            """
            unique_values = set()
            # Iterate over all available indices to ensure all classes are captured.
            for dataset_id, sample_id in self.indices:
                mask_path = Path(self.mask_data[dataset_id][sample_id]).resolve()
                if not mask_path.exists():
                    raise FileNotFoundError(f"Mask file not found: {mask_path}")
                try:
                    mask = tifffile.imread(str(mask_path))
                except Exception as e:
                    raise RuntimeError(f"Failed to open mask with tifffile: {e}")
                # If the mask has multiple channels, take the first channel.
                if mask.ndim > 2:
                    mask = mask[..., 0]
                # For binary segmentation (self.num_classes == 1), normalize to 0 and 1.
                if self.num_classes == 1:
                    unique_vals = np.unique(mask)
                    mask = (mask - unique_vals.min()) / (unique_vals.max() - unique_vals.min())
                unique_values.update(np.unique(mask))
                
                if len(unique_values) >= self.num_classes:
                    break
        
            sorted_values = sorted(unique_values)
            
            # For binary segmentation, explicitly set output to [0, 1]
            if self.num_classes == 1:
                sorted_values = [0, 1]
            # If ignoring background, remove the first (assumed background) value.
            elif self.ignore_background and len(sorted_values) > 1:
                sorted_values.pop(0)
            
            return sorted_values

    def _weights_calc(self, mask, temperature=1.0, epsilon=1e-6):
        weights = np.full(self.num_classes, epsilon, dtype=np.float32)  # Prevent 0s
        unique_classes_in_mask, counts = np.unique(mask, return_counts=True)
     
        total_pixels = np.sum(counts)
     
        class_freqs = {
            val: freq / total_pixels
            for val, freq in zip(unique_classes_in_mask, counts)
            if val in self.class_values
        }
     
        for i, class_value in enumerate(self.class_values):
            freq = class_freqs.get(class_value, epsilon)
            weight = 1.0 / (freq + epsilon)
            weights[i] = weight
     
        # Temperature scaling via power
        weights = np.power(weights, 1.0 / temperature)
        weights = weights / np.sum(weights)  # Normalize to sum to 1
     
        return torch.tensor(weights, dtype=torch.float32)

    def map_mask_to_indices(self, mask):
        """
        Remaps the mask pixel values to sequential class indices based on self.class_values.
        Any pixel that does not match a value in self.class_values is assigned an ignore index (-1).
        """
        if self.num_classes == 1:
            return (mask > 0).astype(np.int64)
        
        remapped_mask = np.full_like(mask, fill_value=-1, dtype=np.int64)
        for idx, class_val in enumerate(self.class_values):
            remapped_mask[mask == class_val] = idx
        return remapped_mask

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding mask by index.
        """
        dataset_id, sample_id = self.indices[idx]
        img_path = Path(self.img_data[dataset_id][sample_id]).resolve()
        if dataset_id in self.data_stats:
            m, s = self.data_stats[dataset_id]
        else:
            m, s = self.data_stats["default"]
        
        if self.inference_mode:
            try:
                img = tifffile.imread(str(img_path))
            except Exception as e:
                raise RuntimeError(f"Failed to open image with tifffile: {e}")
            
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).contiguous() / 255
            patches = self.extract_patches(img_tensor.numpy())
            processed_patches = []
            for i in range(patches.shape[0]):
                patch = patches[i]
                patch_tensor = torch.tensor(patch).unsqueeze(0)
                patch_resized = nn_func.interpolate(patch_tensor, size=(self.img_res, self.img_res),
                                                    mode="bilinear", align_corners=False).squeeze()
                img_normalized = torchvision.transforms.functional.normalize(patch_resized, mean=m, std=s).float()
                processed_patches.append(img_normalized)
            return processed_patches, dataset_id, str(img_path)

        try:
            img = tifffile.imread(str(img_path))
        except Exception as e:
            raise RuntimeError(f"Failed to open image with tifffile: {e}")
        mask_path = Path(self.mask_data[dataset_id][sample_id]).resolve()
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        try:
            mask = tifffile.imread(str(mask_path))
        except Exception as e:
            raise RuntimeError(f"Failed to open mask with tifffile: {e}")
        
        if mask.ndim > 2:
            mask = mask[..., 0]
        
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        
        assert img.shape[:2] == mask.shape, (
            f"Mismatch in dimensions: Image {img.shape} vs Mask {mask.shape} for {img_path}"
        )
        
        img, mask = self.get_valid_crop(img, mask, threshold=0.8, max_attempts=20)
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).contiguous() / 255
        remapped_mask = self.map_mask_to_indices(mask)
        mask_tensor = torch.from_numpy(remapped_mask).contiguous().long()
        
        if self.num_classes > 1:
            weights = self._weights_calc(remapped_mask) if self.weights else torch.zeros(self.num_classes)
        else:
            weights = torch.zeros(self.num_classes)
        
        img_resized = nn_func.interpolate(img_tensor.unsqueeze(0), size=(self.img_res, self.img_res),
                                          mode="bilinear", align_corners=False).squeeze()
        mask_resized = nn_func.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0).float(),
                                           size=(self.img_res, self.img_res), mode="nearest").squeeze().long()
        
        if torch.rand(1).item() < self.p:
            img_resized = torchvision.transforms.functional.hflip(img_resized)
            mask_resized = torchvision.transforms.functional.hflip(mask_resized)
        if torch.rand(1).item() < self.p:
            img_resized = torchvision.transforms.functional.vflip(img_resized)
            mask_resized = torchvision.transforms.functional.vflip(mask_resized)
        
        img_normalized = torchvision.transforms.functional.normalize(img_resized, mean=m, std=s).float()
        
        return img_normalized, mask_resized, weights

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.indices)
