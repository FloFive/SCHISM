import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from Hyperparameters import Hyperparameters  # Ensure the import path is correct
from resUNet_pytorch import ResUNet  # Ensure the import path is correct
from util import Util  # Ensure the import path is correct


def load_model(weights_path, hyperparams):
    """
    Load the model with the specified weights.

    Args:
        weights_path (str): Path to the model weights file.
        hyperparams (Hyperparameters): Hyperparameters for initializing the model.

    Returns:
        ResUNet: The initialized model with loaded weights.
    """
    model = ResUNet(**hyperparams.__dict__)  # Initialize the model with hyperparameters
    model.load_weights(weights_path)  # Load the model weights
    model.eval()  # Set the model to evaluation mode
    return model


def preprocess_image(image_path, scaler):
    """
    Load and preprocess the image for inference.

    Args:
        image_path (str): Path to the input image.
        scaler: Scaler object for normalizing the image.

    Returns:
        np.ndarray: Preprocessed image as a numpy array.
    """
    try:
        image = Image.open(image_path).convert("RGB")  # Load the image in RGB format
        image = np.array(image)  # Convert the image to a numpy array
        image = scaler.transform(image.reshape(-1, 1)).reshape(image.shape)  # Apply the scaler
        return image
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        raise


def predict(model, image):
    """
    Perform prediction on the given image.

    Args:
        model (ResUNet): The model to use for prediction.
        image (np.ndarray): The input image.

    Returns:
        np.ndarray: The predicted mask.
    """
    with torch.no_grad():  # Disable gradient calculation
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # Convert image to tensor
        output = model.model(image_tensor)  # Perform prediction
        predicted_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # Get the predicted class
        return predicted_mask


def display_results(image, mask, preds):
    """
    Display the input image, ground truth mask, and predicted mask.

    Args:
        image (torch.Tensor): The input image.
        mask (torch.Tensor): The ground truth mask (not used in inference).
        preds (torch.Tensor): The predicted mask.
    """
    # Convert tensors to numpy for visualization
    image_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)  # Convert image to numpy and correct dimensions
    mask_np = mask.squeeze().cpu().numpy() if mask is not None else None  # Convert ground truth mask to numpy
    preds_np = preds.squeeze().cpu().numpy()  # Convert predicted mask to numpy

    # Print shapes and value ranges
    print("Image shape:", image_np.shape)
    print("Mask shape:", mask_np.shape if mask_np is not None else "N/A")
    print("Predicted shape:", preds_np.shape)
    print("Image min/max:", image_np.min(), image_np.max())
    print("Predicted mask min/max:", preds_np.min(), preds_np.max())

    # Create subplots to display the image, mask, and predicted mask
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the input image
    axes[0].imshow(image_np)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Plot the ground truth mask if available
    if mask_np is not None:
        axes[1].imshow(mask_np, cmap='gray')
        axes[1].set_title('Ground Truth Mask')
    else:
        axes[1].set_title('Ground Truth Mask (Not Available)')
    axes[1].axis('off')

    # Plot the predicted mask
    axes[2].imshow(preds_np, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')

    plt.show()


def run_inference(image_path, weights_path, hyperparams_path):
    """
    Manage the inference process.

    Args:
        image_path (str): Path to the input image.
        weights_path (str): Path to the model weights file.
        hyperparams_path (str): Path to the hyperparameters file.
    """
    # Load hyperparameters
    try:
        hyperparams = Hyperparameters.load_from_ini(hyperparams_path)
    except Exception as e:
        print(f"Error loading hyperparameters: {e}")
        return

    # Load the model
    try:
        model = load_model(weights_path, hyperparams)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize the utility for preprocessing
    util = Util(stackImage=[image_path], isInference=True, scaler=hyperparams.scaler)

    # Preprocess the image
    try:
        image = preprocess_image(image_path, util.scaler)
        # Perform prediction
        predicted_mask = predict(model, image)
        # Display the results
        # Pass None for the mask as it is not available
        display_results(torch.from_numpy(image), torch.from_numpy(None), torch.from_numpy(predicted_mask))
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return


# The file can be imported without executing any code
