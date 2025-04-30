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
                 num_classes=None, img_res=560, crop_size=(224, 224), p=0.5, inference_mode=False, ignore_background=True,  data_augmentation=False, aug_brightness=0, aug_angle = 15, aug_translate = [10, 10], aug_scale = 1.0, aug_shear = [10, 10]):
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
        self.data_augmentation = data_augmentation
        self.aug_brightness=aug_brightness
        self.angle=aug_angle
        self.translate=aug_translate 
        self.scale=aug_scale
        self.shear=aug_shear               
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

        i = torch.randint(0, h - th + 1, size=(1,)).item() #  generates a random integer between 0 and h - th, to start the crop (starting row), .item() is for convert the tensor to the integer
        j = torch.randint(0, w - tw + 1, size=(1,)).item() #  generates a random integer between 0 and h - th, to start the crop (starting column), .item() is for convert the tensor to the integer

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
        center_i = (h - th) // 2  # Computes the starting row for a center crop, Floor division, ensures we get an integer index
        center_j = (w - tw) // 2  # Computes the starting column for a center crop, Floor division, ensures we get an integer index
    
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
            pad_height = (patch_h - height % patch_h) % patch_h    # If H is not a multiple of C_H, we need to pad in the height direction.
            pad_width = (patch_w - width % patch_w) % patch_w      # If W is not a multiple of C_W, we need to pad in the width direction.

            # Correct padding tuple for a 3D image (H, W, C)
            padding = [(0, pad_height), (0, pad_width), (0, 0)]  # Pad along height and width, no padding for channels

            # Pad the image
            img_np = np.pad(img_np, padding, mode='constant', constant_values=0)

        # Transpose back to [C, H, W] after padding
        img_np = np.transpose(img_np, (2, 0, 1)).squeeze()   
        # 1) Before this step, the image is in [H, W, C] format (Height, Width, Channels), np.transpose(img_np, (2, 0, 1)) changes the order of axes; 2) .squeeze() function removes axes (dimensions) with size 1 from an array. It does not remove other dimensions.
        patches = patchify(img_np, (img_np.shape[0], patch_h, patch_h), step=patch_h)  # Split the image into patches using patchify()
        patches = patches.reshape(-1, img_np.shape[0], patch_h, patch_w)  # The -1 in the reshape function tells NumPy to automatically compute the size of that dimension. In this case, it calculates how many patches you have in total.
        #  resulting shape after reshaping (num_patches, channels, patch_height, patch_width)
        return patches
    
    def _compute_class_values(self):
        """
        Computes unique class values from all available masks.

        Returns:
            list: Sorted list of unique class values in the dataset.
        """
        unique_values = set()  # set() creates a set, which is an unordered collection of unique elements, it is used here to collect unique class values from multiple masks without allowing duplicates.

        for dataset_id, sample_id in self.indices[:10]: 
            mask_path = self.mask_data[dataset_id][sample_id]
            mask = np.array(Image.open(mask_path).convert("L"))  # converted to grayscale (mode "L"), now the mask is then converted to a NumPy array for easier manipulation.
            
            # Normalize the mask to [0, 1] only if binary classification (self.num_classes == 1)
            if self.num_classes == 1:
                unique_vals = np.unique(mask)
                mask = (mask - unique_vals.min()) / (unique_vals.max() - unique_vals.min())  # Scale to [0, 1]
            
            # Update the set of unique class values
            unique_values.update(np.unique(mask))
        
        # Convert the set of unique values to a sorted list
        sorted_values = sorted(unique_values)

        # Handle binary classification (ensure only 0 and 1): binary classification may be represented with a single class, typically with the values 0 (background) and 1 (foreground). 
        if self.num_classes == 1:  # num_classes = 1 indicates that there is one active class (the foreground class, for example), and the background is implicitly represented as 0 (often not explicitly included in the class count).
            sorted_values = [0, 1]  # Binary classification, classes should be 0 and 1
        elif self.ignore_background and len(sorted_values) > 1:
            sorted_values.pop(0)  # Remove the background class (usually class 0) It removes the first element (0) of the list and returns that element, the element of the index in original list will be removed
        
        return sorted_values
        
    def _weights_calc(self, mask, temperature=50.0):
        # This function calculates class weights based on the frequency of each class in the provided mask. These weights can be used in weighted loss functions to give more importance to less frequent classes during training.
        # It is a specific mask that is passed to this function when it's called — usually during training, and typically one patch or crop at a time.
        # Initialize weights with 0 for each class
        weights = np.zeros(self.num_classes, dtype=np.float32)
        
        # Get the unique classes present in the mask
        unique_classes_in_mask = np.unique(mask)

        # Iterate over all the classes defined in self.class_values
        for i, class_value in enumerate(self.class_values):
            if class_value in unique_classes_in_mask:
                # Compute weight for present class 
                class_frequency = np.sum(mask == class_value)  # Count the occurrences of class_value in mask
                class_weight = 1 / (class_frequency + 1e-8)  # Inverse of frequency: rare classes get higher weights (since 1 / frequency makes the weight larger for small frequencies). 
                weights[i] = class_weight
            else:
                # Set weight to 0 for potential missing classes (happens when crop size is small)
                weights[i] = 0

        # Apply temperature scaling if needed
        weights = weights / temperature 
        
        # Normalize weights so they sum to 1
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights / total_weight  # Normalize weights to sum to 1: The weights are normalized to sum to 1 for proper scaling.
        
        # Ensure weights are non-negative
        weights = np.maximum(weights, 0)
        
        # Convert to a tensor
        return torch.tensor(weights)

    def __getitem__(self, idx):  ## Serves as the core function for retrieving an image and its corresponding mask from the dataset
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
        #Step 1: Get paths using the dataset index
        dataset_id, sample_id = self.indices[idx]
        img_path = self.img_data[dataset_id][sample_id]

        #Step 2: Load normalization stats # why normalization: Centers data around zero (subtract mean); Standardizes different scales (divide by std); Prevents large values from dominating small values; Improves training speed and stability
        if dataset_id in self.data_stats:
            m, s = self.data_stats[dataset_id]
        else:
            m, s = self.data_stats["default"]    
        #Step 3: Inference Mode, in this case, only process the image data, instead of the mask data
        if self.inference_mode:
            img = np.array(Image.open(img_path).convert("RGB"))
            img_tensor = torch.from_numpy(img).float()  # .float() ensures pixel values are of type float32, which is required for most deep learning models.
            img_tensor = img_tensor.permute(2, 0, 1).contiguous() / 255 
            # .permute can rearranges the dimensions from [H, W, C] to [C, H, W] because PyTorch expects image tensors in the format of [Channels, Height, Width]... .permute() does not move the actual data in memory. Some operations (like .view() or resizing) need the tensor to be physically stored in a contiguous block of memory.
            img_normalized = torchvision.transforms.functional.normalize(img_tensor, mean=m, std=s)
            patches = self.extract_patches(img_normalized)
            processed_patches = []
        #Step 3.1: Processes each patch from a full image, resizes it, and adds it to the output list.
            for i in range(patches.shape[0]):   # patches.shape[0] is the number of patches extracted.
                patch = patches[i]
                patch_tensor = torch.tensor(patch).unsqueeze(0) # I now have a batch (a group of patches or images) dimension, and this batch contains 1 image. Pytorch uses batch in processing.
                patch_resized = nn_func.interpolate(patch_tensor, size=(self.img_res, self.img_res),
                                                    mode="bicubic", align_corners=False).squeeze()  #reshaping and resizing image tensors to make them ready for models, Neural Networks Expect Fixed Input Size, Batch Processing Requires Same Size
                processed_patches.append(patch_resized)  # A list of resized image patches, ready to be fed into a neural network, these patches are the actual input to your model for making predictions.

            return processed_patches, dataset_id, img_path
        #Step 4: Training Mode: If you're not in inference mode, you go on to load the mask
        mask_path = self.mask_data[dataset_id][sample_id]
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))            
        # Step 5: Check dimensions
        assert img.shape[:2] == mask.shape, (
            f"Mismatch in dimensions: Image {img.shape} vs Mask {mask.shape} for {img_path}"
        )   # The assert statement checks whether a condition is True. If the condition is False, it raises an AssertionError and optionally prints the message you provide.   
            # img.shape = (height, width, 3); mask.shape = (height, width)
        #Step 6: Cropping
        img, mask = self.get_valid_crop(img, mask, threshold=0.8, max_attempts=20)  # It will use the values you pass when calling the function: threshold=0.8; max_attempts=20 → overrides the default max_attempts=10.  So defaults are only used when you don’t explicitly pass a value.
        #Step 7: Convert to torch tensors and normalize image
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).contiguous() / 255
        #Step 8: Handle the mask + weight map: Purpose: converting the raw mask (which is a NumPy array loaded from an image file) into a PyTorch tensor that
        if self.num_classes > 1:  # self.num_classes > 1 → Multiclass Segmentation
            mask_tensor = torch.from_numpy(mask).contiguous() / 255  #  mask is likely saved as an 8-bit grayscale image, where classes are represented as pixel intensities from 0 to 255; Dividing by 255 rescales values to [0, 1], which is easier to normalize or interpret when training.
            weights = self._weights_calc(mask)   #  This dynamically computes class weights based on the current mask, using inverse class frequencies.
        else:  # self.num_classes == 1 → Binary Segmentation
            unique_vals = np.unique(mask)
            mask = (mask - unique_vals.min()) / (unique_vals.max() - unique_vals.min()) # Here you assume the mask only contains foreground (1) and background (0). But sometimes binary masks are saved as [0, 255], so you rescale manually.
            mask = mask.astype(np.int64)   # This is important because loss functions (like CrossEntropyLoss) expect class indices as long (int64) tensors — not floats.
            mask_tensor = torch.from_numpy(mask).contiguous()
            weights = torch.zeros(self.num_classes) # Avoid setting to None  # In the binary case, you’re not dynamically computing weights, probably because: You only care about foreground vs. background

        img_resized = nn_func.interpolate(img_tensor.unsqueeze(0), size=(self.img_res, self.img_res),
                                          mode="bicubic", align_corners=False).squeeze()
        mask_resized = nn_func.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0).float(), size=(self.img_res, self.img_res),
                                        mode="nearest").squeeze()

        if torch.rand(1).item() < self.p:
            img_resized = torchvision.transforms.functional.hflip(img_resized)
            mask_resized = torchvision.transforms.functional.hflip(mask_resized)

        if torch.rand(1).item() < self.p:
            img_resized = torchvision.transforms.functional.vflip(img_resized)
            mask_resized = torchvision.transforms.functional.vflip(mask_resized)
        if self.data_augmentation:
            # Affine Transformation (angle, translate, shear)
            img_resized = func_torch.affine(img_resized, angle=self.aug_angle, translate=self.aug_translate,
                                    scale=self.aug_scale, shear=self.aug_shear)
            mask_affine = mask_resized.unsqueeze(0).unsqueeze(0).float()  # For mask: shape [H, W] or [1, H, W] → add batch and channel dimensions # mask_resized is 2D or 3D (e.g., shape [H, W] or [1, H, W]), but affine() expects a 4D tensor of shape [N, C, H, W].
            mask_affine = func_torch.affine(mask_affine, angle=self.aug_angle, translate=self.aug_translate,
                                    scale=self.aug_scale, shear=self.aug_shear,
                                    interpolation=func_torch.InterpolationMode.NEAREST)            
            mask_resized = mask_affine.squeeze(0).squeeze(0) # Remove batch and channel dims → shape [H, W]
            if self.aug_brightness > 0:
                brightness_factor = torch.empty(1).uniform_(1 - self.aug_brightness, 1 + self.aug_brightness).item()
                img_resized = func_torch.adjust_brightness(img_resized, brightness_factor)
        img_normalized = torchvision.transforms.functional.normalize(img_resized, mean=m, std=s).float()
 
        if self.num_classes > 1:
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