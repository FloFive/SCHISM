# import numpy as np
# from PIL import Image

# class SaveMask:

#     def __init__(self, mask_tensor,save_path):
#         """
#         Saves a segmentation mask to a file.

#         Args:
#             mask_tensor (torch.Tensor): The mask tensor to be saved.
#         """
#         self.mask_tensor = mask_tensor
#         self.save_path = save_path

#     def save(self):
#         mask_array = self.mask_tensor.cpu().numpy().astype(np.uint8)
#         mask_image = Image.fromarray(mask_array)
#         mask_image.save(self.save_path)


import numpy as np
from PIL import Image
import os

class SaveMask:
    def __init__(self, mask_tensor, save_path):
        """
        Saves a segmentation mask to a file.

        Args:
            mask_tensor (torch.Tensor): The mask tensor to be saved.
            save_path (str): Path to save the mask image.
        """
        self.mask_tensor = mask_tensor
        self.save_path = save_path

    def save(self):
        # Convert to numpy and remove unnecessary dimensions
        mask_array = self.mask_tensor.detach().cpu().numpy()
        print(f"[DEBUG] Original mask shape: {mask_array.shape}, dtype: {mask_array.dtype}")
        
         # If it’s one-hot encoded (C, H, W), convert to class index (H, W)
        if mask_array.ndim == 3 and mask_array.shape[0] <= 10:  # assuming num_classes ≤ 10
            mask_array = np.argmax(mask_array, axis=0)  # now shape (H, W)
            print(f"[DEBUG] After argmax: {mask_array.shape}, dtype: {mask_array.dtype}")
        else:
            mask_array = np.squeeze(mask_array)
            print(f"[DEBUG] After squeeze: {mask_array.shape}, dtype: {mask_array.dtype}")

        if mask_array.ndim != 2:
            raise ValueError(f"[ERROR] Final mask array shape should be 2D. Got: {mask_array.shape}")


        # Normalize to [0, 255] if not uint8 already
        if mask_array.dtype != np.uint8:
            mask_array = mask_array - np.min(mask_array)
            if np.max(mask_array) > 0:
                mask_array = mask_array / np.max(mask_array) * 255
            mask_array = mask_array.astype(np.uint8)

        # Make sure the directory exists
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        # Save mask
        Image.fromarray(mask_array).save(self.save_path)
        print(f"[✓] Saved mask: {self.save_path}")