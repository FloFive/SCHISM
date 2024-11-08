# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:33:49 2022
@author: florent.brondolo
"""
import random
import numpy as np
from numpy import histogram
from numpy.core.multiarray import ndarray
from skimage.exposure import equalize_adapthist, match_histograms, rescale_intensity, adjust_log
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from .stack import Stack
from code.Backup.ProgressBar import ProgressBar
import matplotlib.pyplot as plt
import skimage.exposure as expo


class Util:
    """
    A utility class for handling image stacks, performing preprocessing, and managing dataset splitting.
    """

    def __repr__(self):
        """
        Returns a string representation of the class.
        Returns:
            str: The name of the class 'Util'.
        """
        return 'Util'

    def __init__(self, **kwargs):
        """
               Initializes the Util object with optional parameters for image stacks, labels, preprocessing functions,
               and training configurations.
               Args:
                   name (str): Name of the instance.
                   numSlice (int): Number of slices per stack to process.
                   image_preprocessing_functions (list of str): List of preprocessing function names.
                   scaler (sklearn.preprocessing object): Scaler object for normalization.
                   validation_split (float): Split ratio for validation data (default: 0.3).
                   isInference (bool): Whether this instance is for inference (default: False).
                   stackImage (list): List of image stacks.
                   stackLabel (list): List of label stacks.
               """
        num_slice_sampled_per_stack = 0
        self.x_train = self.y_train = self.x_test = self.y_test = np.zeros([0, 0, 0, 0])
        self.name = kwargs.get('name', "")
        self.numSlice = kwargs.get('numSlice', None)
        self.image_preprocessing_functions = kwargs.get('image_preprocessing_functions', None)
        self.scaler = kwargs.get('scaler', None)
        self.validation_split = kwargs.get('validation_split', 0.3)
        self.inference = kwargs.get('isInference', False)

        # Rest of the initialization...
        if "stackImage" in kwargs and "stackLabel" not in kwargs:
            stack_image_list = kwargs['stackImage']
            self.image_type = stack_image_list[0].getImageType()
            if self.numSlice is not None:
                num_slice_sampled_per_stack = int(self.numSlice / len(stack_image_list))
                num_slice_provided = True
            else:
                num_slice_provided = False
            self.num_class = stack_image_list[0].getNumClass()
            selected_slice_image = []
            pb = ProgressBar(len(stack_image_list), txt=repr(self) + '- Loading ' + self.name)
            for stack_image_tmp in stack_image_list:
                if num_slice_provided:
                    if num_slice_sampled_per_stack <= len(stack_image_tmp.getListSlice()):
                        index_slice = random.sample(range(0, stack_image_tmp.getStackSize()),
                                                    num_slice_sampled_per_stack)
                        selected_slice_image += stack_image_tmp.getSliceFromPosition(index_slice)
                    else:
                        selected_slice_image += stack_image_tmp.getListSlice()
                else:
                    selected_slice_image += stack_image_tmp.getListSlice()
                pb += 1
            self.stack_image = Stack(
                imageType=self.image_type,
                width=stack_image_list[0].getWidth(),
                height=stack_image_list[0].getHeight(),
                isSegmented=stack_image_list[0].getIsSegmented(),
                isSliceListSupplied=True,
                selectedFiles=selected_slice_image,
                channel=stack_image_list[0].getChannel(),
                numClass=self.num_class
            )
            self.numSlice = self.stack_image.get_stack_size()
        elif "stackImage" in kwargs and "stackLabel" in kwargs:
            def unique_class(unique_classes):
                """
                Creates a mapping of unique classes to integer indices.
                Args:
                    unique_classes (list): A list of unique class labels.
                Returns:
                    dict: A dictionary mapping each unique class to a corresponding integer index.
                """
                class_mapping = {}
                for idx, cls in enumerate(unique_classes):
                    class_mapping[cls] = idx
                return class_mapping

            def normalize_class(stacked_sequence_normalize, class_mapping):
                """
                Normalizes a stacked sequence of classes using a provided class mapping.
                Args:
                    stacked_sequence_normalize (ndarray): A multidimensional array where
                                                            the last dimension contains class labels.
                    class_mapping (dict): A dictionary mapping class labels to integer indices.
                Returns:
                    ndarray: The normalized array with class labels replaced by their corresponding indices.
                """
                for j in range(stacked_sequence_normalize.shape[-1]):
                    stacked_sequence_normalize[..., j] = np.vectorize(class_mapping.get)(
                        stacked_sequence_normalize[..., j])
                return stacked_sequence_normalize

            stack_image_list = kwargs['stackImage']
            stack_label_list = kwargs['stackLabel']
            first_image_type = stack_image_list[0].getImageType()
            for stack in stack_image_list:
                if stack.getImageType() != first_image_type:
                    raise Exception(repr(self) + ' class - Image types must be the same between stacks')
                else:
                    self.image_type = first_image_type
            if self.numSlice is not None:
                num_slice_sampled_per_stack = int(self.numSlice / len(stack_image_list))
                num_slice_provided = True
            else:
                num_slice_sampled_per_stack = 0
                num_slice_provided = False
            if stack_image_list[0].getNumClass() == stack_label_list[0].getNumClass():
                self.num_class = stack_image_list[0].getNumClass()
                selected_slice_image = []
                selected_slice_label = []
                pb = ProgressBar(len(stack_image_list) + 4, txt=repr(self) + '- Loading ' + self.name)
                for stack_image_tmp, stack_label_tmp in zip(stack_image_list, stack_label_list):
                    if num_slice_provided:
                        if num_slice_sampled_per_stack <= len(stack_image_tmp.getListSlice()):
                            index_slice = random.sample(range(0, stack_image_tmp.getStackSize()),
                                                        num_slice_sampled_per_stack)
                            selected_slice_image += stack_image_tmp.getSliceFromPosition(index_slice)
                            selected_slice_label += stack_label_tmp.getSliceFromPosition(index_slice)
                        else:
                            selected_slice_image += stack_image_tmp.getListSlice()
                            selected_slice_label += stack_label_tmp.getListSlice()
                    else:
                        selected_slice_image += stack_image_tmp.getListSlice()
                        selected_slice_label += stack_label_tmp.getListSlice()
                    pb += 1
                self.stack_image = Stack(
                    width=stack_image_list[0].getWidth(),
                    height=stack_image_list[0].getHeight(),
                    isSegmented=stack_image_list[0].getIsSegmented(),
                    isSliceListSupplied=True,
                    selectedFiles=selected_slice_image,
                    channel=stack_image_list[0].getChannel()
                )
                # Class count
                stacked_sequence = np.stack(selected_slice_label, axis=-1)
                pb += 1
                unique_classes_temp = np.unique(stacked_sequence)
                pb += 1
                class_mapping_temp = unique_class(list(unique_classes_temp))
                pb += 1
                stacked_sequence = normalize_class(stacked_sequence, class_mapping_temp)

                # retreive unique normalized class from the class_mapping dict
                self.unique_class = []
                keys = list(class_mapping_temp.keys())
                keys.sort()
                for key in keys:
                    self.unique_class.append(class_mapping_temp[key])
                pb += 1
                list_mask = []
                for i in range(stacked_sequence.shape[-1]):
                    image = stacked_sequence[..., i]
                    list_mask.append(image)
                self.stack_label = Stack(
                    width=stack_label_list[0].getWidth(),
                    height=stack_label_list[0].getHeight(),
                    isSegmented=stack_label_list[0].getIsSegmented(),
                    isSliceListSupplied=True,
                    selectedFiles=list_mask,
                    channel=stack_label_list[0].getChannel()
                )
                self.numClass = len(class_mapping_temp)
                self.stack_image.set_num_class(self.num_class)
                self.numSlice = self.stack_image.get_stack_size()
                self.class_frequency = self.weights_calc()
                list_size = len(kwargs['stackImage'])
                for i in range(list_size):
                    kwargs["stackImage"][i].setnumClass(len(class_mapping_temp))
                    kwargs['stackLabel'][i].setnumClass(len(class_mapping_temp))
            else:
                raise Exception(repr(self) + ' class - Only list of stacks are accepted as inputs')

    def get_num_slice(self):
        """
               Returns the number of slices in the stack.
               Returns:
                   int: Number of slices.
               """
        return self.numSlice

    def get_x_train(self):
        """
              Returns the training images.
              Returns:
                  ndarray: Training images.
              """
        return self.x_train

    def get_y_train(self):
        """
               Returns the training labels.
               Returns:
                   ndarray: Training labels.
               """
        return self.y_train

    def get_x_test(self):
        """
               Returns the test images.
               Returns:
                   ndarray: Test images.
               """
        return self.x_test

    def get_y_test(self):
        """
              Returns the test labels.
              Returns:
                  ndarray: Test labels.
              """
        return self.y_test

    def get_num_class(self):
        """
               Returns the number of classes in the dataset.
               Returns:
                   int: Number of classes.
               """
        return self.numClass

    def get_stack_image(self):
        """
              Returns the stack of images.
              Returns:
                  Stack: Image stack object.
              """
        return self.stack_image

    def get_image_type(self):
        """
               Returns the image type of the stack.
               Returns:
                   str: Image type.
               """
        return self.image_type

    def get_stack_label(self):
        """
              Returns the stack of labels.
              Returns:
                  Stack: Label stack object.
              """
        return self.stack_label

    def get_validation_split(self):
        """
               Returns the validation split ratio.
               Returns:
                   float: Validation split ratio.
               """
        return self.validation_split

    def get_image_preprocessing_functions(self):
        """
               Returns the list of image preprocessing functions.
               Returns:
                   list: List of preprocessing function names.
               """
        return self.image_preprocessing_functions

    def get_class_frequency(self):
        """
               Returns the frequency of each class in the dataset.
               Returns:
                   list: Class frequencies.
               """
        return self.class_frequency

    def get_unique_class(self):
        """
              Returns the unique class labels.
              Returns:
                  list: Unique class labels.
              """
        return self.unique_class

    def weights_calc(self):
        """
               Calculates class weights based on class frequencies.
               Returns:
                   list: Normalized class weights.
               """
        unique_classes, class_counts = np.unique(self.stack_label.get_list_slice(), return_counts=True)
        sorted_classes = unique_classes[np.argsort(class_counts)]
        class_weights = 1.0 / class_counts
        class_weight_dict = {class_label: weight for class_label, weight in zip(sorted_classes, class_weights)}
        total_weight = sum(class_weight_dict.values())
        class_weights = [weight / total_weight for weight in class_weights]
        return class_weights

    @staticmethod
    def clahe(dataset):
        """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to a dataset of images.
        Args:
            dataset (list of ndarray): The dataset of images.
        Returns:
            ndarray: CLAHE adjusted images.
        """
        preview = True  # Control the preview display
        output = []  # List to store the CLAHE-adjusted images
        for img in dataset:
            # Normalize the image to a range of [-1, 1]
            img_normalized = (img.astype(np.float32) - np.min(img)) / (np.max(img) - np.min(img)) * 2.0 - 1.0
            matched = equalize_adapthist(img_normalized)  # Apply CLAHE
            output.append(matched)  # Append the adjusted image to the output list
            if preview:  # Show preview for the first image
                fig, axes = plt.subplots(1, 2)
                axes[0].imshow(img, cmap='gray')
                axes[0].set_title('Original', fontsize=10)
                axes[1].imshow(matched, cmap='gray')
                axes[1].set_title('CLAHE adjusted', fontsize=10)
                plt.show()
                preview = False  # Only show the preview once
        return np.array(output)  # Convert the output list to a NumPy array and return it

    def adjust_log(self, preview_log=False):
        """
           Applies logarithmic adjustment to a dataset of images.
           Args:
               self (list of ndarray): Dataset of images to adjust.
               preview_log (bool): If True, shows a comparison of original and adjusted images.
           Returns:
               ndarray: Log adjusted images.
           """
        output = []  # Create an empty list to store the equalized cubes
        for img in self:  # Iterate over each cube in the input dataset
            matched = adjust_log(img, 3)  # Apply histogram equalization using equalize_hist() function
            output.append(matched)  # Append the equalized cube to the output list
            if preview_log:  # If test is True, display a comparison of original and equalized cube images
                fig, axes = plt.subplots(1, 2)
                axes[0].imshow(img, cmap='gray')
                axes[0].set_title('Original', fontsize=10)
                axes[1].imshow(matched, cmap='gray')
                axes[1].set_title('Log adjusted', fontsize=10)
                plt.show()
        return np.array(output)  # Convert the output list to a NumPy array and return it


    # Function to perform gamma adjustment on a dataset of 3D cubes
    def adjust_gamma(dataset, gamma_value=1.0):
        """
           Performs gamma adjustment on a dataset of 3D cubes.
           Parameters:
               dataset (list or numpy.ndarray): The dataset containing the image cubes.
               gamma_value (float): The gamma value to use for the adjustment. Default is 1.0.
           Returns:
               numpy.ndarray: An array containing the gamma-adjusted image cubes.
           """
        preview = True  # A boolean variable to control display of test images
        output = []  # Create an empty list to store the equalized cubes
        for img in dataset:  # Iterate over each cube in the input dataset
            matched = expo.adjust_gamma(img, gamma_value)  # Apply histogram equalization using equalize_hist() function
            output.append(matched)  # Append the equalized cube to the output list
            if preview:  # If test is True, display a comparison of original and equalized cube images
                fig, axes = plt.subplots(1, 2)
                axes[0].imshow(img, cmap='gray')
                axes[0].set_title('Original', fontsize=10)
                axes[1].imshow(matched, cmap='gray')
                axes[1].set_title('Gamma adjusted (γ = {})'.format(gamma_value), fontsize=10)
                plt.show()
                preview = False
        return ndarray(output)  # Convert the output list to a NumPy array and return it


    def normalize_histograms(dataset):
        """
        Normalize the histograms of a dataset using the best representative image.
        Args:
            dataset (list of ndarray): A list of input images.
        Returns:
            list of ndarray: The normalized images.
        """
        print("normalize_histograms")  # Should be removed after deployment

    def select_best_reference(dataset_reference):
        """
        Select the best representative image from a dataset.
        Args:
            dataset_reference (list of ndarray): A list of input images.
        Returns:
            ndarray: The best representative image.
        """
        # Calculate the overall histogram for the entire dataset
        overall_histogram = np.zeros(256, dtype=np.float64)  # Change dtype to float64 for precision
        for pic in dataset_reference:
            overall_histogram += histogram(pic, nbins=256)[0]  # Update histogram calculation
        # Normalize the overall histogram
        overall_histogram = overall_histogram / np.sum(overall_histogram)  # Explicitly use np.sum
        # Compute representativeness scores for each image
        representativeness_scores = []
        for pic in dataset_reference:
            img_histogram = histogram(pic, nbins=256)[0]
            img_histogram = img_histogram / np.sum(img_histogram)  # Explicitly use np.sum
            # Calculate the similarity score based on histogram similarity
            similarity_score = np.sum(np.minimum(overall_histogram, img_histogram))
            representativeness_scores.append(similarity_score)
        # Find the index of the image with the highest representativeness score
        # best_index = int(np.argmax(representativeness_scores))  # Ensure best_index is a native Python int
        best_index = int(np.amax(representativeness_scores))  # Ensure best_index is a native Python int
        # Return the best representative image
        return dataset_reference[best_index]

        # Select the best representative image
        best_reference = select_best_reference(dataset)
        preview = True  # A boolean variable to control display of test images
        normalized_images = []
        for img in dataset:
            # Perform histogram matching separately for each channel (if multichannel)
            if img.ndim == 3:  # Check if the image is multichannel (color)
                matched_channels = []
                for channel in range(img.shape[2]):
                    matched = match_histograms(img[:, :, channel], best_reference[:, :, channel])
                    matched_channels.append(matched)
                matched = np.stack(matched_channels, axis=2)
            else:
                # For single-channel images, perform histogram matching directly
                matched = match_histograms(img, best_reference)
            normalized_images.append(matched)
            if preview:
                fig, axes = plt.subplots(1, 2)
                axes[0].imshow(img, cmap='gray')
                axes[0].set_title('Original', fontsize=10)
                axes[0].axis('off')
                axes[1].imshow(matched, cmap='gray')
                axes[1].set_title('Normalized', fontsize=10)
                axes[1].axis('off')
                plt.show()
                preview = False
        return np.array(normalized_images)


    # Function to perform contrast stretching on a dataset of 3D cube
    def contrast_stretching(dataset):
        """
       Apply contrast stretching to a dataset of 3D images (or cubes) by adjusting intensity.
       Args:
           dataset (list of ndarray): A list of input images.
       Returns:
           list of ndarray: The contrast-stretched images.
       """
        preview = True  # A boolean variable to control display of test images
        print("contrast_stretching")  # Should be removed after deployment
        for i, img in enumerate(dataset):
            # Ensure img is a Numpy array
            img = np.array(img)
            p2, p98 = np.percentile(img, (5, 95))  # Calculate the 5th and 95th percentiles
            # Update the dataset with the contrast-stretched image
            # Rescale intensity based on the percentiles
            matched = rescale_intensity(img, in_range=(p2, p98))
            dataset[i] = matched  # Update the dataset with the contrast-stretched image
            if preview:  # Display original and contrast-stretched images for the first slice in the first batch
                fig, axes = plt.subplots(1, 2)
                axes[0].imshow(img, cmap='gray')
                axes[0].set_title('Original', fontsize=10)
                axes[1].imshow(matched, cmap='gray')
                axes[1].set_title('Contrast stretched', fontsize=10)
                plt.show()
                preview = False
        return dataset  # Return the modified dataset

    # noinspection PyUnreachableCode
    def load_data(self):
        """
        Load and preprocess image data, including normalization, splitting into training and test sets, and applying
        image preprocessing functions. This method handles both training and inference cases.
        Returns:
        tuple: Training and test image data (x_train, y_train, x_test, y_test).
        """
        if hasattr(self, 'stackLabel'):  # Training case
            x = np.zeros([self.stackImage.getStackSize(), self.stackImage.getHeight(), self.stackImage.getWidth(),
                          self.stackImage.getChannel()])
            y: ndarray = np.zeros(
                [self.stackLabel.getStackSize(), self.stackLabel.getHeight(), self.stackLabel.getWidth(),
                 self.stackLabel.getChannel()])
            for j in range(self.stackImage.getStackSize()):
                x[j, :, :, :] = self.stackImage.get_list_slice()[j]
                y[j, :, :, :] = self.stackLabel.get_list_slice()[j]
            # Image treatment
            # Call the functions using their names from the array
            if self.image_preprocessing_functions:
                for func_name in self.image_preprocessing_functions:
                    func = getattr(self, func_name, None)
                    if func is not None and callable(func):
                        x = func(x)
            if self.numClass > 2:  # one hot encode multiclass dataset
                encoder = OneHotEncoder(sparse_output=False)
                y_encoded = encoder.fit_transform(y.reshape(-1, 1))
                y = y_encoded.reshape((self.numSlice, self.stackLabel.getHeight(), self.stackLabel.getWidth(),
                                       self.numClass))
                del encoder, y_encoded
            else:  # clipping binary data between 0 and 1
                y = np.stack([np.minimum(np.maximum(arr, 0), 1) for arr in y], axis=0).astype(np.float32)
            if self.inference:
                x_train_transformed = (
                    self.scaler.fit_transform(x.reshape(-1, 1)).reshape(x.shape))  # Apply the scaler to the image data
                self.x_test = None
                self.x_train = x_train_transformed
                self.y_test = None
                self.y_train = y
                del x, y
            else:
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.validation_split)
                del x, y
                scaler = MinMaxScaler(feature_range=(0, 1),
                                      clip=True)  # Initialize a MinMaxScaler to scale the image data
                x_train_transformed = (scaler.fit_transform(x_train.reshape(-1, 1)).reshape(
                    x_train.shape))  # Apply the scaler to training data
                x_test_transformed = (
                    scaler.transform(x_test.reshape(-1, 1)).reshape(x_test.shape))  # Apply the scaler to test data
                self.scaler = scaler
                self.x_train = x_train_transformed  # Use the transformed training data
                self.x_test = x_test_transformed  # Use the transformed testing data
                self.y_train = y_train
                self.y_test = y_test
            return self.x_train, self.y_train, self.x_test, self.y_test
        else:  # Inference case
            x = np.zeros([self.stack_image.get_stack_size(),
                          self.stack_image.get_height(),
                          self.stack_image.get_width(),
                          self.stack_image.get_channel()])

            for j in range(self.stack_image.get_stack_size()):
                x[j, :, :, :] = self.stack_image.get_list_slice()[j]
            # Image treatment

            # Call the functions using their names from the array
            if self.image_preprocessing_functions:
                for func_name in self.image_preprocessing_functions:
                    func = getattr(self, func_name, None)
                    if func is not None and callable(func):
                        x = func(x)
            x_transformed = (
                self.scaler.fit_transform(x.reshape(-1, 1)).reshape(x.shape))  # Apply the scaler to the image data
            self.x_train = x_transformed
            self.x_test = None
            self.y_test = None
            self.y_train = None
            return self.x_train, self.y_train, self.x_test, self.y_test
