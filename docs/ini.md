
## [Model]
Defines the model architecture and hyperparameters.

- `model_type` supports the following models:
  - `UnetVanilla`:  A simplistic UNet (preset by default) --> [more info here](https://github.com/FloFive/SCHISM/blob/main/docs/UnetVanilla.md)
  - `UnetSegmentor`: A more robust UNet for more complex tasks (our adaptation of Liang et al. 2022) --> [more info here](https://github.com/FloFive/SCHISM/blob/main/docs/UnetSegmentor.md)
  - `DINOv2`: A powerful pretrained vision transformer for very precise tasks (our implementation; see [this link](https://arxiv.org/abs/2407.18100) for the related paper) --> [more info here](https://github.com/FloFive/SCHISM/blob/main/docs/DINOv2.md)

- `num_classes` is mandatory.
- Optional parameters have predefined default values that vary depending on the specific network. Please look at the network's documentation or file for detailed guidance on parameter configuration. These are:
  - `k_size`
  - `activation`
  - `n_block`
  - `channels` 

  
More models will be added in the future.

## [Optimizer]
Specifies the optimizer. Choices include:

- `Adagrad`
- `Adam` (preset by default)
- `AdamW`
- `NAdam`
- `RMSprop`
- `RAdam`
- `SGD`

These options are derived from the `torch.optim` library, and parameters, if any, should match the library's documentation.

## [Scheduler]
Specifies the learning rate scheduler. Options include:

- `LRScheduler`
- `LambdaLR`
- `MultiplicativeLR`
- `StepLR`
- `MultiStepLR`
- `ConstantLR` (preset by default)
- `LinearLR`
- `ExponentialLR`
- `PolynomialLR`
- `CosineAnnealingLR`
- `SequentialLR`
- `ReduceLROnPlateau`
- `CyclicLR`
- `OneCycleLR`
- `CosineAnnealingWarmRestarts`

These options are derived from `torch.optim.lr_scheduler`. Parameters, if any, should match the library's documentation.

## [Loss]
Specifies the loss. Choices include:

- `CrossEntropyLoss` (preset by default)
- `BCEWithLogitsLoss`
- `NLLLoss`

All parameters in the `torch.nn` library should be configurable as described in the official documentation. Regarding `ignore_index` and `weight`, please see the details below:
- `ignore_background`: Set whether the background class be ignored (`True`) during training and inference, or not (`False`). If set to `True`, `num_classes` should be decreased by one. Should be kept at `False` for binary data. This would correspond to the ignore_index attribute of the torch.nn background.
- `weights`: If set to `True`, class weights will be computed for each present class. By default, this is `False`. Class weighting is not supported for binary classification and will be automatically disabled, even if explicitly set to `True`.


These options are derived from `torch.nn`. Parameters, if any, should match the library's documentation.

## [Training]
Configures training parameters.

- Batch Size (`batch_size`), Validation Split (`val_split`), and Epochs (`epochs`): These parameters should be carefully adjusted based on your dataset and hardware constraints. By default, they are set as follows:
  - `batch_size`: 8
  - `val_split`: 0.8 (80% of the dataset used for training, 20% for validation)
  - `epochs`: 10
    
- Evaluation Metrics (`metrics`): The model performance can be assessed using one or more of the following metrics:
  - `Jaccard` (preset by default)
  - `F1-score`
  - `Accuracy`
  - `Precision`
  - `Recall`
  - Additionally, setting metrics to include `ConfusionMatrix` will generate a confusion matrix plot at the end of training.
    
- Early Stopping (`early_stopping`): When enabled (`True`), early stopping prevents overfitting by monitoring the loss value. If the loss remains unchanged for a certain number of epochs, training is stopped automatically. The patience period (i.e., the number of epochs with stable loss before stopping) is set to 20% of the total epoch count. For example, if epochs=50, early stopping will trigger after 10 consecutive epochs without improvement. This function can either be ignored or set to `False`.

## [Data]
Configures data handling.

- `crop_size`: Size of image crops. The default value is 224px.
- `img_res`: Resolution to resize crops during training and inference. The default value is 560px.
- `num_samples`: Number of samples to use. The default value is 500 samples.

During training, images are split into crops of user-defined size (`crop_size`), then resized to `img_res` to ensure compatibility with various machines and GPUs while preserving detail. The same process is applied during inference, with patches reassembled into the original image.

