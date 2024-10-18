"""
            ███████╗   ██████╗  ██╗  ██╗  ██╗  ███████╗  ███╗   ███╗
            ██╔════╝  ██╔════╝  ██║  ██║  ██║  ██╔════╝  ████╗ ████║
            ███████╗  ██║       ███████║  ██║  ███████╗  ██╔████╔██║
            ╚════██║  ██║       ██╔══██║  ██║  ╚════██║  ██║╚██╔╝██║
            ███████║  ╚██████╗  ██║  ██║  ██║  ███████║  ██║ ╚═╝ ██║
            ╚══════╝   ╚═════╝  ╚═╝  ╚═╝  ╚═╝  ╚══════╝  ╚═╝     ╚═╝
            Semantic Classification of High-resolution Imaging for Scanned Materials
"""

# Import necessary libraries
import sys
import os
import random
import colorsys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from joblib import load
from tensorflow.python.keras.models import model_from_json
from classes.stack import Stack
from classes.util import Util
from matplotlib.colors import ListedColormap
import cv2  # Import OpenCV
import configparser  # Import configparser for .ini file handling

# Set TensorFlow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Create a CLAHE object
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Define project directories
project_dir = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(project_dir, "code"))
runs_dir = os.path.join(project_dir, 'runs')
data_dir = os.path.join(project_dir, 'data')

# Define constants
IMAGE_TYPE = {'scanner': 0, 'image': 1}  # not to be changed

# Function to generate colors
def generate_colors(n_class):
    n_class = max(n_class, 1)
    hsv_colors = [(x / n_class, 1, 1) for x in range(n_class)]
    rgb_colors = [tuple(int(c * 255) for c in colorsys.hsv_to_rgb(*hsv)) for hsv in hsv_colors]
    return rgb_colors

# Function to remove spaces from a list
def remove_spaces_from_list(word_list):
    return [word.replace(" ", "") for word in word_list]

# Function to get parameters from log file
def get_parameters_from_log_file(log_file_path):
    parameters = {}
    with open(log_file_path, "r") as log_file:
        for line in log_file:
            if 'Informational inputs' in line:
                break
            line = line.strip()
            if line and not line.startswith("#"):
                line_split = line.split(" = ")
                if len(line_split) >= 2:
                    parameter_name, parameter_value = line_split[0].strip(), line_split[1].strip()
                    if parameter_name in ['image_preprocessing_functions', 'metrics', 'early_stopping']:
                        parameter_value = parameter_value.strip('[]').replace('\'', '').split(',')
                    parameters[parameter_name] = parameter_value
    return parameters

# Hyperparameters class
class Hyperparameters:
    def __init__(self, **kwargs):
        """
        Initializes the Hyperparameters instance with optional parameters.
        """
        self.img_side_length = kwargs.get('img_side_length', 512)
        self.num_sample = kwargs.get('num_sample', 1000)
        self.imgtype = kwargs.get('imgtype', "scanner")
        self.pretrained = kwargs.get('pretrained', True)
        self.backbone = kwargs.get('backbone', 'resnet50')
        self.image_preprocessing_functions = kwargs.get('image_preprocessing_functions', [])
        self.metrics = kwargs.get('metrics', ["CategoricalCrossentropy, OneHotMeanIoU"])
        self.featuremaps = kwargs.get('featuremaps', 16)
        self.epochs = kwargs.get('epochs', 20)
        self.val_split = kwargs.get('val_split', 0.8)
        self.displaySummary = kwargs.get('displaySummary', True)
        self.maxNorm = kwargs.get('maxNorm', 3)
        self.learningRate = kwargs.get('learningRate', 1e-4)
        self.batchNorm = kwargs.get('batchNorm', True)
        self.batch_size = kwargs.get('batch_size', 3)
        self.save_model = kwargs.get('save_model', True)
        self.dropOut = kwargs.get('dropOut', True)
        self.dropOutRate = kwargs.get('dropOutRate', 0.4)
        self.L2 = kwargs.get('L2', 1e-4)
        self.early_stopping = kwargs.get('early_stopping', [])
        self.loss_early_stopping = kwargs.get('loss_early_stopping', False)
        self.patience = kwargs.get('patience', 25)
        self.run_name = kwargs.get('run_name', "default_run")

    def __repr__(self):
        return f"Hyperparameters({self.__dict__})"

    def save_to_ini(self, file_path):
        """Saves hyperparameters to an .ini file."""
        config = configparser.ConfigParser()
        config['Hyperparameters'] = {k: str(v) for k, v in self.__dict__.items()}
        with open(file_path, 'w') as configfile:
            config.write(configfile)

    @classmethod
    def load_from_ini(cls, file_path):
        """Loads hyperparameters from an .ini file."""
        config = configparser.ConfigParser()
        config.read(file_path)
        params = {k: v for k, v in config['Hyperparameters'].items()}
        return cls(**params)

# Hyperparameters & parameters (manual setting)
hyperparams = Hyperparameters(
    img_side_length=512,
    num_sample=1000,
    imgtype="scanner",
    pretrained=True,
    backbone='resnet50',
    image_preprocessing_functions=[CLAHE],
    metrics=["CategoricalCrossentropy, OneHotMeanIoU"],
    featuremaps=16,
    epochs=20,
    val_split=80 / 100,
    displaySummary=True,
    maxNorm=3,
    learningRate=1e-4,
    batchNorm=True,
    batch_size=3,
    save_model=True,
    dropOut=True,
    dropOutRate=0.4,
    L2=1e-4,
    early_stopping=[],
    loss_early_stopping=False,
    patience=25,
    run_name="bentheimer-multiclass-run-06-11-2023--10h-23m-57s"
)

# Save hyperparameters to .ini file
hyperparams.save_to_ini('hyperparameters.ini')

# Load parameters from log file
file_dict = get_parameters_from_log_file(os.path.join(runs_dir, hyperparams.run_name, 'logs.txt'))

# Update hyperparameters from log file
hyperparams.img_width, hyperparams.img_height = int(file_dict['img_width']), int(file_dict['img_height'])
hyperparams.imageType = int(file_dict['imageType'])
hyperparams.num_sample = int(file_dict['numSample'])
hyperparams.featuremaps = int(file_dict["featuremaps"])
hyperparams.epochs = int(file_dict["epochs"])
hyperparams.val_split = float(file_dict["val_split"])
hyperparams.maxNorm = int(file_dict["MaxNorm"])
hyperparams.learningRate = float(file_dict["learningRate"])
hyperparams.batchNorm = bool(file_dict["batchNorm"])
hyperparams.batch_size = int(file_dict["batch_size"])
hyperparams.metrics = file_dict['metrics']
hyperparams.early_stopping = file_dict['early_stopping']
hyperparams.save_model = bool(file_dict["save_model"])
hyperparams.dropOut = bool(file_dict["dropOut"])
hyperparams.dropOutRate = float(file_dict["dropoutRate"])
hyperparams.L2 = float(file_dict["L2"])
hyperparams.patience = int(file_dict["patience"])
hyperparams.pretrained = bool(file_dict["pretrained"])
hyperparams.loss_early_stopping = bool(file_dict['loss_early_stopping'])
hyperparams.image_preprocessing_functions = file_dict["image_preprocessing_functions"]

# Volume import
name_stack = ["test_alhammadi"]  # @param {type:"raw"}
name_stack = remove_spaces_from_list(name_stack)

stackImage = []
stackLabel = []

for image in name_stack:
    mask = image + '/masks'
    image = image + '/images'
    A_image = Stack(name='images',
                    imageType=hyperparams.imageType,
                    isSegmented=False,
                    width=hyperparams.img_width,
                    height=hyperparams.img_height,
                    numSlice=hyperparams.num_sample,
                    path=os.path.join(data_dir, image))
    stackImage.append(A_image)

    A_mask = Stack(name='masks',
                   isSegmented=True,
                   width=hyperparams.img_width,
                   height=hyperparams.img_height,
                   stackImage=A_image,
                   selectedFiles=A_image.get_selected_files(),
                   path=os.path.join(data_dir, mask))
    stackLabel.append(A_mask)

util = Util(name="batch test",
            validation_split=hyperparams.val_split,
            image_preprocessing_functions=hyperparams.image_preprocessing_functions,
            stackImage=stackImage,
            stackLabel=stackLabel)
x_train, y_train, x_test, y_test = util.loadData()

# 2nd visual check (optional)
print("Xtrain - Min:", np.min(x_train))
print("Xtrain - Max:", np.max(x_train))
print("Xtest - Min:", np.min(x_test))
print("Xtest - Max:", np.max(x_test))
print("Ytrain - Min:", np.min(y_train))
print("Ytrain - Max:", np.max(y_train))
print("Ytest - Min:", np.min(y_test))
print("Ytest - Max:", np.max(y_test))

N_class = stackImage[0].get_num_class()
custom_colors = generate_colors(N_class)
print(custom_colors)

custom_cmap = ListedColormap(custom_colors)

# Visualize training and testing data
i = random.randint(0, len(x_test) - 1)
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(x_train[i], cmap='gray')
axes[0, 0].set_title('Xtrain', fontsize=15)
axes[0, 0].axis('off')
if N_class <= 2:
    axes[0, 1].imshow(y_train[i])
else:
    axes[0, 1].imshow(np.argmax(y_train[i], axis=-1))
axes[0, 1].set_title('Ytrain', fontsize=15)
axes[0, 1].axis('off')
axes[1, 0].imshow(x_test[i], cmap='gray')
axes[1, 0].set_title('Xtest', fontsize=15)
axes[1, 0].axis('off')
if N_class <= 2:
    axes[1, 1].imshow(y_test[i])
else:
    axes[1, 1].imshow(np.argmax(y_test[i], axis=-1))
axes[1, 1].set_title('Ytest', fontsize=15)
axes[1, 1].axis('off')

# Compute class counts
def compute_class_counts(labels):
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    print("Class Counts:")
    for class_label, count in zip(unique_classes, class_counts):
        print(f"Class {class_label}: {count} samples")

compute_class_counts(stackLabel[0].get_list_slice())

# Model training
neuralNetwork = ResUNet(pathLogDir=runs_dir,
                        featuremaps=hyperparams.featuremaps,
                        data=util,
                        epochs=hyperparams.epochs,
                        batch_size=hyperparams.batch_size,
                        learningRate=hyperparams.learningRate,
                        L2=hyperparams.L2,
                        batchNorm=hyperparams.batchNorm,
                        maxNorm=hyperparams.maxNorm,
                        dropOut=hyperparams.dropOut,
                        dropOutRate=hyperparams.dropOutRate,
                        metrics=hyperparams.metrics,
                        early_stopping=hyperparams.early_stopping,
                        loss_early_stopping=hyperparams.loss_early_stopping,
                        patience=hyperparams.patience,
                        padding="same",
                        pretrained=hyperparams.pretrained,
                        backbone=hyperparams.backbone,
                        save_model=hyperparams.save_model,
                        displaySummary=hyperparams.displaySummary)

neuralNetwork.set_model()
neuralNetwork.run()

# Settings for prediction
run_name = "binary-pred-demo"  # @param {type:"string"}
num_sample = 10  # @param {type:"integer"}
name_stack = ["bentheimer_testing"]  # @param {type:"raw"}
name_stack = remove_spaces_from_list(name_stack)
real_inference = False  # @param {type:"boolean"}

file_dict = get_parameters_from_log_file(os.path.join(runs_dir, run_name, 'logs.txt'))
imageType = int(file_dict['imageType'])
img_width, img_height = int(file_dict['img_width']), int(file_dict['img_height'])
scaler = load(os.path.join(runs_dir, run_name, 'std_scaler_image.bin'))  # Data scaler loading
model_path = os.path.join(runs_dir, run_name, 'weights')
json_file = open(os.path.join(model_path, 'weights.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(os.path.join(model_path, 'weights.h5'))
pretrained = bool(file_dict["pretrained"])
numClass = int(file_dict["numClass"])
image_preprocessing_functions = file_dict["image_preprocessing_functions"]

# Scanner loading & preprocessing
if real_inference:
    stackImage = []
    for images in name_stack:
        images = images + '/images'
        ImagesPred = Stack(name='images',
                           imageType=imageType,
                           isSegmented=False,
                           width=img_width,
                           height=img_height,
                           numSlice=num_sample,
                           path=os.path.join(data_dir, images))
        stackImage.append(ImagesPred)

    util2 = Util(name="batch prediction",
                 scaler=scaler,
                 image_preprocessing_functions=image_preprocessing_functions,
                 stackImage=stackImage)

    img_pred, y_train, x_test, y_test = util2.loadData()
    if pretrained and util2.getImageType() == 1:
        img_pred = np.concatenate([img_pred] * 3, axis=-1)
    y_pred_threshold = generate_prediction(loaded_model, img_pred, numClass)
else:
    stackImage = []
    stackLabel = []
    for image in name_stack:
        mask = image + '/masks'
        image = image + '/images'
        A_image = Stack(name='images',
                        imageType=imageType,
                        isSegmented=False,
                        width=img_width,
                        height=img_height,
                        numSlice=num_sample,
                        path=os.path.join(data_dir, image))
        stackImage.append(A_image)
        A_mask = Stack(name='masks',
                       isSegmented=True,
                       width=img_width,
                       height=img_height,
                       stackImage=A_image,
                       selectedFiles=A_image.get_selected_files(),
                       path=os.path.join(data_dir, mask))
        stackLabel.append(A_mask)

    util_testing = Util(name="batch testing",
                        isInference=True,
                        scaler=scaler,
                        image_preprocessing_functions=image_preprocessing_functions,
                        stackImage=stackImage,
                        stackLabel=stackLabel)

    img_pred, maskTesting, null_variable1, null_variable2 = util_testing.loadData()
    numClass = util_testing.getNumClass()

    if pretrained and util_testing.getImageType() == 1:
        img_pred = np.concatenate([img_pred] * 3, axis=-1)

    y_pred_threshold = generate_prediction(loaded_model, img_pred, numClass)

# Prediction display
if real_inference:
    i = random.randint(0, len(img_pred) - 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(img_pred[i], cmap='gray')
    axes[0].set_title('Original', fontsize=15)
    axes[0].axis('off')

    if numClass > 2:
        predicted_cmap = plt.cm.get_cmap('viridis', numClass)
        axes[1].imshow(y_pred_threshold[i], cmap=predicted_cmap, vmin=0, vmax=numClass - 1)
        axes[1].set_title('Predicted', fontsize=15)
        axes[1].axis('off')
    else:
        binary_predictions = y_pred_threshold[i].squeeze()
        axes[1].imshow(binary_predictions, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Predicted', fontsize=15)
        axes[1].axis('off')
else:
    i = random.randint(0, len(img_pred) - 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_pred[i], cmap='gray')
    axes[0].set_title('Original', fontsize=15)
    axes[0].axis('off')

    if numClass > 2:
        ground_truth_mask = maskTesting[i].argmax(axis=-1)
        ground_truth_cmap = plt.cm.get_cmap('viridis', numClass)
        axes[1].imshow(ground_truth_mask, cmap=ground_truth_cmap, vmin=0, vmax=numClass - 1)
        axes[1].set_title('Ground Truth', fontsize=15)
        axes[1].axis('off')

        predicted_cmap = plt.cm.get_cmap('viridis', numClass)
        color_image = y_pred_threshold[i].reshape(img_pred.shape[1], img_pred.shape[2], 1)
        axes[2].imshow(color_image, cmap=predicted_cmap, vmin=0, vmax=numClass - 1)
        axes[2].set_title('Predicted', fontsize=15)
        axes[2].axis('off')
    else:
        binary_ground_truth = maskTesting[i].squeeze()
        axes[1].imshow(binary_ground_truth, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Ground Truth', fontsize=15)
        axes[1].axis('off')

        binary_predictions = y_pred_threshold[i].squeeze()
        axes[2].imshow(binary_predictions, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Predicted', fontsize=15)
        axes[2].axis('off')