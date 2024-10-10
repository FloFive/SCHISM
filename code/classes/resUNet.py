# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:45:14 2022

@author: florent.brondolo
"""
import os
from datetime import datetime
import keras
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models as sm
import tensorflow as tf
import tensorflow.python.keras as keras
import tensorflow.python.keras.layers as layers
from joblib import dump
from keras import backend as kera
from tensorflow.python.keras.layers import
    Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, BatchNormalization, Activation
from tensorflow.python.losses import CategoricalFocalCrossentropy
from tensorflow.python.models import Model
from tensorflow.python.optimizers import Adam
from tensorflow.python.client import device_lib

from util import Util

tf.keras.

class ResUNet:
    """
    A class representing a Residual U-Net model for image segmentation.

    Methods:
        __repr__(): Returns a string representation of the class.
        __init__(**kwargs): Initializes the ResUNet class with given parameters.
        get_train_data(): Returns the training data.
        get_test_data(): Returns the testing data.
        get_available_gpu(): Returns a list of available GPU devices.
        get_validation_split(): Returns the validation split.
        data_generator(seed, batch_size, is_train_set, for_visualisation=False):
            Generates batches of data for training or testing.
        get_model(): Returns the current model.
        weighted_categorical_crossentropy(): Returns a weighted categorical crossentropy loss function.
        weighted_binary_crossentropy(): Returns a weighted binary crossentropy loss function.
        set_model(): Configures and sets up the U-Net model architecture.
        run(): Compiles and trains the model, with options for data augmentation and early stopping.
    """

    def __repr__(self):
        """Returns a string representation of the class."""
        return 'ResUNet'

    def __init__(self, **kwargs):
        """Initializes the ResUNet class with given parameters."""
        self.model = None
        self.validation_split = None
        if kwargs['pathLogDir'] is not None:
            self.pathLogDir = kwargs['pathLogDir']
        else:
            raise Exception(repr(self) + ' class - pathLogDir variable must me provided')

        self.file_name = "run-" + datetime.now().strftime("%d-%m-%Y--%Hh-%Mm-%Ss")
        self.logdir = os.path.join(self.pathLogDir, self.file_name)

        self.FILE_TXT = '-----------------------------------' + '\r'
        self.FILE_TXT = self.FILE_TXT + 'runName = ' + self.file_name + '\r'
        self.FILE_TXT = self.FILE_TXT + '-----------------------------------' + '\r'

        if "featuremaps" in kwargs:
            self.featuremaps = int(kwargs['featuremaps'])
        else:
            self.featuremaps = 8
        self.FILE_TXT = self.FILE_TXT + "\nfeaturemaps = " + str(self.featuremaps)

        if "data" in kwargs:
            if isinstance(kwargs['data'], Util):
                self.util = kwargs['data']
                self.x_train = self.util.getXtrain()
                self.y_train = self.util.getYtrain()
                self.x_test = self.util.getXtest()
                self.y_test = self.util.getYtest()
                self.num_class = self.util.getNumClass()
                self.FILE_TXT = self.FILE_TXT + "\nnumClass = " + str(self.num_class)
                self.FILE_TXT = self.FILE_TXT + "\nnumSample = " + str(self.util.getNumSlice())
                self.FILE_TXT = self.FILE_TXT + "\nval_split = " + str(self.util.get_validation_split())
                self.FILE_TXT = self.FILE_TXT + "\nimage_preprocessing_functions = ["
                if len(self.util.getImagePreprocessingFunctions()) > 0:
                    marker = len(self.util.getImagePreprocessingFunctions()) - 1
                    i = 0
                    for function in self.util.getImagePreprocessingFunctions():
                        if len(self.util.getImagePreprocessingFunctions()) == 1:
                            self.FILE_TXT = self.FILE_TXT + function
                        elif len(self.util.getImagePreprocessingFunctions()) > 1:
                            if marker == i:
                                self.FILE_TXT = self.FILE_TXT + function
                            else:
                                self.FILE_TXT = self.FILE_TXT + function + ","
                        i += 1
                self.FILE_TXT = self.FILE_TXT + "]"
                self.FILE_TXT = self.FILE_TXT + "\nimageType = " + str(self.util.getImageType())
            else:
                raise Exception(repr(self) + ' class - training and testing dataset must be provided')

        if "epochs" in kwargs:
            self.epochs = int(kwargs['epochs'])
        else:
            self.epochs = 50
        self.FILE_TXT = self.FILE_TXT + "\nepochs = " + str(self.epochs)

        if "batch_size" in kwargs:
            self.batch_size = int(kwargs['batch_size'])
        else:
            self.batch_size = 8
        self.FILE_TXT = self.FILE_TXT + "\nbatch_size = " + str(self.batch_size)

        if "learningRate" in kwargs:
            self.learning_rate = float(kwargs['learningRate'])
        else:
            self.learning_rate = 1e-5
        self.FILE_TXT = self.FILE_TXT + "\nlearningRate = " + str(self.learning_rate)

        if "L2" in kwargs:
            self.L2 = float(kwargs['L2'])
        else:
            self.L2 = 1e-5
        self.FILE_TXT = self.FILE_TXT + "\nL2 = " + str(self.L2)

        if "batchNorm" in kwargs:
            self.batch_norm = bool(kwargs['batchNorm'])
            self.FILE_TXT = self.FILE_TXT + "\nbatchNorm = " + str(self.batch_norm)
        else:
            self.batch_norm = False
            self.FILE_TXT = self.FILE_TXT + "\nbatchNorm = none"

        if "maxNorm" in kwargs:
            self.max_norm = int(kwargs['maxNorm'])
        else:
            self.max_norm = 1
        self.FILE_TXT = self.FILE_TXT + "\nMaxNorm = " + str(self.max_norm)

        if "dropOut" in kwargs:
            self.drop_out = bool(kwargs['dropOut'])
            self.FILE_TXT = self.FILE_TXT + "\ndropOut = " + str(self.drop_out)
        else:
            self.drop_out = False
            self.FILE_TXT = self.FILE_TXT + "\ndropOut = none"

        if self.drop_out:
            if "dropoutRate" in kwargs:
                self.dropout_rate = float(kwargs['dropoutRate'])
            else:
                self.dropout_rate = 0.1
            self.FILE_TXT = self.FILE_TXT + "\ndropoutRate = " + str(self.dropout_rate)
        else:
            self.dropout_rate = None
            self.FILE_TXT = self.FILE_TXT + "\ndropoutRate = none"

        if "dataGeneration" in kwargs:
            self.data_generation = bool(kwargs['dataGeneration'])
        else:
            self.data_generation = False
        self.FILE_TXT = self.FILE_TXT + "\ndataGeneration = " + str(self.data_generation)

        if "additional_augmentation_factor" in kwargs:
            self.additional_augmentation_factor = kwargs['additional_augmentation_factor']
        else:
            self.additional_augmentation_factor = 2
        self.FILE_TXT = self.FILE_TXT + "\nadditional_augmentation_factor = " + str(self.additional_augmentation_factor)

        if "patience" in kwargs:
            self.patience = int(kwargs['patience'])
        else:
            self.patience = 5
        self.FILE_TXT = self.FILE_TXT + "\npatience = " + str(self.patience)

        if "padding" in kwargs:
            if kwargs['padding'] == "same" or kwargs['padding'] == "valid":
                self.padding = kwargs['padding']
            else:
                self.padding = "same"
        else:
            self.padding = "same"
        self.FILE_TXT = self.FILE_TXT + "\npadding = " + str(self.padding)

        if "pretrained" in kwargs:
            self.pretrained = bool(kwargs['pretrained'])
            if "backbone" in kwargs:
                self.backbone = str(kwargs['backbone'])
                self.FILE_TXT = self.FILE_TXT + "\nbackbone= " + str(self.backbone)
        else:
            self.pretrained = False
        self.FILE_TXT = self.FILE_TXT + "\npretrained = " + str(self.pretrained)

        if "img_height" in kwargs:
            self.img_height = int(kwargs['img_height'])
        else:
            self.img_height = self.x_train.shape[1]
        self.FILE_TXT = self.FILE_TXT + "\nimg_height = " + str(self.img_height)

        if "img_width" in kwargs:
            self.img_width = int(kwargs['img_width'])
        else:
            self.img_width = self.x_train.shape[2]
        self.FILE_TXT = self.FILE_TXT + "\nimg_width = " + str(self.img_width)
        metrics_early_stopping = []
        metrics_early_stopping_tmp = []
        if "metrics" in kwargs:
            if "early_stopping" in kwargs:
                metrics_early_stopping = kwargs['early_stopping']
                metrics_early_stopping_tmp = []

            self.FILE_TXT = self.FILE_TXT + "\nmetrics = ["
            self.metrics = kwargs['metrics']
            if len(self.metrics) > 0:
                marker = len(kwargs['metrics']) - 1
                i = 0
                metrics = []
                for metric_name in self.metrics:
                    if len(kwargs['metrics']) == 1:
                        self.FILE_TXT = self.FILE_TXT + str(metric_name)
                    elif len(kwargs['metrics']) > 1:
                        if marker == i:
                            self.FILE_TXT = self.FILE_TXT + str(metric_name)
                        else:
                            self.FILE_TXT = self.FILE_TXT + str(metric_name) + ","
                    if metric_name == 'OneHotIoU' or metric_name == 'OneHotMeanIoU':
                        # Pass num_classes for OneHotIoU and OneHotMeanIoU metrics
                        if metric_name == 'OneHotIoU':
                            target_class_ids = np.array(self.util.getUniqueClass(), dtype=np.int32)  # Convert to int32
                            metric_instance = globals()[metric_name](num_classes=self.num_class,
                                                                     target_class_ids=target_class_ids)
                        else:
                            metric_instance = globals()[metric_name](num_classes=self.num_class)
                    else:
                        # For other metrics, create instances without num_classes
                        metric_instance = globals()[metric_name]()

                    metrics.append(metric_instance)

                    if "early_stopping" in kwargs:
                        if len(metrics_early_stopping) > 0:
                            if metric_name in metrics_early_stopping:
                                metrics_early_stopping_tmp.append('val_' + metric_instance.name)
                    i += 1
                self.metrics = metrics
                if "early_stopping" in kwargs:
                    self.early_stopping = metrics_early_stopping_tmp
            else:
                self.metrics.append('accuracy')
                self.FILE_TXT = self.FILE_TXT + "accuracy"
        else:
            self.metrics.append('accuracy')
            self.FILE_TXT = self.FILE_TXT + "accuracy"
        self.FILE_TXT = self.FILE_TXT + "]"

        if "early_stopping" in kwargs:
            self.FILE_TXT = self.FILE_TXT + "\nearly_stopping = ["
            if len(self.early_stopping) > 0:
                marker = len(self.early_stopping) - 1
                i = 0
                for early_stopping in self.early_stopping:
                    if len(self.early_stopping) == 1:
                        self.FILE_TXT = self.FILE_TXT + str(early_stopping)
                    elif len(self.early_stopping) > 1:
                        if marker == i:
                            self.FILE_TXT = self.FILE_TXT + str(early_stopping)
                        else:
                            self.FILE_TXT = self.FILE_TXT + str(early_stopping) + ","
                    i += 1
            self.FILE_TXT = self.FILE_TXT + "]"

        if "loss_early_stopping" in kwargs:
            self.loss_early_stopping = kwargs['loss_early_stopping']
            self.FILE_TXT = self.FILE_TXT + "\nloss_early_stopping = " + str(self.loss_early_stopping)

        if "save_model" in kwargs:
            self.save_model = kwargs['save_model']
        else:
            self.save_model = False
        self.FILE_TXT = self.FILE_TXT + "\nsave_model = " + str(self.save_model)

        self.FILE_TXT = self.FILE_TXT + "\n------- Informational inputs -------"

        self.num_gpus = self.get_available_gpu()
        self.FILE_TXT = self.FILE_TXT + "\nGPU(s) = " + str(self.num_gpus)

        if "loss" in kwargs:
            self.loss = kwargs['loss']
        else:
            if self.num_class == 2:  # Binary
                self.loss = self.weighted_binary_crossentropy()
                # self.loss = BinaryCrossentropy(from_logits=True)
            else:  # Multiclass
                self.loss = CategoricalFocalCrossentropy()
                # self.loss = self.weighted_categorical_crossentropy()
                # self.loss = CategoricalCrossentropy(from_logits=True)
        self.FILE_TXT = self.FILE_TXT + "\nLoss function = " + str(self.loss)

        if "displaySummary" in kwargs:
            self.displaySummary = kwargs['displaySummary']
        else:
            self.displaySummary = True

    def get_train_data(self):
        """Returns the training data."""
        return self.x_train, self.y_train

    def get_test_data(self):
        """Returns the testing data."""
        return self.x_test, self.y_test

    @staticmethod
    def get_available_gpu():
        """Returns a list of available GPU devices."""
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    def get_validation_split(self):
        """Returns the validation split."""
        return self.validation_split

    def data_generator(self, seed, batch_size, is_train_set, for_visualisation=False):
        """
        Generates batches of data for training or testing.

        Args:
            seed (int): Random seed for data shuffling.
            batch_size (int): Number of samples per batch.
            is_train_set (bool): Whether the data is for training.
            for_visualisation (bool): Whether the data is for visualization.

        Yields:
            Tuple: A batch of input and output data.
        """
        if is_train_set:
            image_datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=15,  # Rotate images up to 15 degrees
                                               width_shift_range=0.1,  # Shift width by up to 10%
                                               height_shift_range=0.1,  # Shift height by up to 10%
                                               horizontal_flip=True,  # Horizontal flip
                                               vertical_flip=True,  # Vertical flip
                                               shear_range=0.2,  # Shear up to 20%
                                               fill_mode='reflect'  # Fill mode
                                               )
            image_datagen.fit(self.x_train)
            gen_x1 = image_datagen.flow(self.x_train, self.y_train, batch_size=batch_size, seed=seed, shuffle=is_train_set)
            image_datagen.fit(self.y_train)
            gen_x2 = image_datagen.flow(self.y_train, self.x_train, batch_size=batch_size, seed=seed, shuffle=is_train_set)
        else:
            image_datagen = keras.preprocessing.image.ImageDataGenerator()
            image_datagen.fit(self.x_train)
            gen_x1 = image_datagen.flow(self.x_train, self.y_train, batch_size=batch_size, seed=seed, shuffle=is_train_set)
            image_datagen.fit(self.y_train)
            gen_x2 = image_datagen.flow(self.y_train, self.x_train, batch_size=batch_size, seed=seed, shuffle=is_train_set)

        while True:
            if for_visualisation:
                x1i = gen_x1.next()
                x2i = gen_x2.next()
                yield [x1i[0], x2i[0]], [x2i[1], x1i[1]]
            else:
                x1i = gen_x1.next()
                x2i = gen_x2.next()
                yield x1i[0], x2i[0]

    def get_model(self):
        """Returns the current model."""
        return self.model

    def weighted_categorical_crossentropy(self):
        """Returns a weighted categorical crossentropy loss function."""
        weights = kera.variable(self.util.getClassFrequency(), dtype='float32')

        def loss(y_true, y_pred):
            # scale predictions so that the class probas of each sample sum to 1
            # y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = kera.clip(y_pred, kera.epsilon(), 1 - kera.epsilon())
            # calc
            loss_in = y_true * kera.log(y_pred) * weights
            loss_in = -kera.sum(loss_in, -1)
            return loss_in

        return loss

    def weighted_binary_crossentropy(
            self):  # taken from https://github.com/huanglau/Keras-Weighted-Binary-Cross-Entropy/blob/master/DynCrossEntropy.py
        """Returns a weighted binary crossentropy loss function."""
        def loss(y_true, y_pred):
            # calculate the binary cross entropy
            bin_crossentropy = kera.binary_crossentropy(y_true, y_pred)
            # apply the weights
            weights = y_true * self.util.getClassFrequency()[1] + (1. - y_true) * self.util.getClassFrequency()[0]
            weighted_bin_crossentropy = weights * bin_crossentropy
            return kera.mean(weighted_bin_crossentropy)

        return loss

    def set_model(self):
        """Configures and sets up the U-Net model architecture."""
        global final

        def batch_norm_activation(x):  # To off Batch Normalization, set BN to False
            """
            Applies batch normalization, ReLU activation, and optional dropout to the input tensor.

            Args:
                x: Input tensor to be normalized and activated.

            Returns:
                Tensor after applying batch normalization, ReLU activation, and optional dropout.
            """
            if self.batch_norm and self.drop_out:
                x = BatchNormalization()(x)
                x = Activation("relu")(x)
                x = Dropout(self.dropout_rate)(x)
            elif self.batch_norm and not self.drop_out:
                x = BatchNormalization()(x)
                x = Activation("relu")(x)
            elif not self.batch_norm and self.drop_out:
                x = Activation("relu")(x)
                x = Dropout(self.dropout_rate)(x)
            else:
                x = Activation("relu")(x)
            return x

        if self.pretrained:

            preprocess_input1 = sm.get_preprocessing(self.backbone)

            # preprocess input
            if self.util.getImageType() == 0:
                self.x_train = np.concatenate([self.x_train] * 3, axis=-1)
                self.x_test = np.concatenate([self.x_test] * 3, axis=-1)
            self.x_train = preprocess_input1(self.x_train)
            self.x_test = preprocess_input1(self.x_test)

            if self.num_class > 2:
                self.model = sm.Unet(self.backbone, encoder_weights='imagenet', classes=self.num_class,
                                     activation='softmax')
            elif self.num_class <= 2:
                self.model = sm.Unet(self.backbone, encoder_weights='imagenet', classes=1, activation='sigmoid')

            if self.displaySummary:
                self.model.summary()

                # if not using pretrained models
        else:
            inputs = keras.layers.Input(shape=(self.img_height, self.img_width, self.x_train.shape[-1]))
            # Architecture borrowed from https://github.com/AhadMomin/semantic-segmentation-digital-rock-physics
            conv1 = Conv2D(self.featuremaps * 1, (3, 3), padding=self.padding,
                           kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                           kernel_constraint=keras.constraints.max_norm(self.max_norm))(inputs)
            conv1 = batch_norm_activation(conv1)
            conv1 = Conv2D(self.featuremaps * 1, (3, 3), padding=self.padding,
                           kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                           kernel_constraint=keras.constraints.max_norm(self.max_norm))(conv1)
            conv1 = batch_norm_activation(conv1)
            pool1 = MaxPooling2D((2, 2))(conv1)

            conv2 = Conv2D(self.featuremaps * 2, (3, 3), padding=self.padding,
                           kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                           kernel_constraint=keras.constraints.max_norm(self.max_norm))(pool1)
            conv22 = batch_norm_activation(conv2)
            shortcut = Conv2D(self.featuremaps * 2, (3, 3), padding=self.padding,
                              kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                              kernel_constraint=keras.constraints.max_norm(self.max_norm))(conv22)
            shortcut = batch_norm_activation(shortcut)
            conv2 = Conv2D(self.featuremaps * 2, (3, 3), padding=self.padding,
                           kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                           kernel_constraint=keras.constraints.max_norm(self.max_norm))(conv22)
            conv2 = batch_norm_activation(conv2)
            conv2 = Conv2D(self.featuremaps * 2, (3, 3), padding=self.padding,
                           kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                           kernel_constraint=keras.constraints.max_norm(self.max_norm))(conv2)
            conv2 = batch_norm_activation(conv2)
            conv2 = layers.add([shortcut, conv2])
            pool2 = MaxPooling2D((2, 2))(conv2)

            conv3 = Conv2D(self.featuremaps * 4, (3, 3), padding=self.padding,
                           kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                           kernel_constraint=keras.constraints.max_norm(self.max_norm))(pool2)
            conv33 = batch_norm_activation(conv3)
            shortcut1 = Conv2D(self.featuremaps * 4, (3, 3), padding=self.padding,
                               kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                               kernel_constraint=keras.constraints.max_norm(self.max_norm))(conv33)
            shortcut1 = batch_norm_activation(shortcut1)
            conv3 = Conv2D(self.featuremaps * 4, (3, 3), padding=self.padding,
                           kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                           kernel_constraint=keras.constraints.max_norm(self.max_norm))(conv33)
            conv3 = batch_norm_activation(conv3)
            conv3 = Conv2D(self.featuremaps * 4, (3, 3), padding=self.padding,
                           kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                           kernel_constraint=keras.constraints.max_norm(self.max_norm))(conv3)
            conv3 = batch_norm_activation(conv3)
            conv3 = layers.add([shortcut1, conv3])
            pool3 = MaxPooling2D((2, 2))(conv3)

            conv4 = Conv2D(self.featuremaps * 8, (3, 3), padding=self.padding,
                           kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                           kernel_constraint=keras.constraints.max_norm(self.max_norm))(pool3)
            conv44 = batch_norm_activation(conv4)
            shortcut2 = Conv2D(self.featuremaps * 8, (3, 3), padding=self.padding,
                               kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                               kernel_constraint=keras.constraints.max_norm(self.max_norm))(conv44)
            shortcut2 = batch_norm_activation(shortcut2)
            conv4 = Conv2D(self.featuremaps * 8, (3, 3), padding=self.padding,
                           kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                           kernel_constraint=keras.constraints.max_norm(self.max_norm))(conv44)
            conv4 = batch_norm_activation(conv4)
            conv4 = Conv2D(self.featuremaps * 8, (3, 3), padding=self.padding,
                           kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                           kernel_constraint=keras.constraints.max_norm(self.max_norm))(conv4)
            conv4 = batch_norm_activation(conv4)
            conv4 = layers.add([shortcut2, conv4])
            pool4 = MaxPooling2D((2, 2))(conv4)

            convm = Conv2D(self.featuremaps * 16, (3, 3), padding=self.padding,
                           kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                           kernel_constraint=keras.constraints.max_norm(self.max_norm))(pool4)
            convm = batch_norm_activation(convm)
            shortcut3 = Conv2D(self.featuremaps * 16, (3, 3), padding=self.padding,
                               kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                               kernel_constraint=keras.constraints.max_norm(self.max_norm))(convm)
            shortcut3 = batch_norm_activation(shortcut3)
            convm = Conv2D(self.featuremaps * 16, (3, 3), padding=self.padding,
                           kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                           kernel_constraint=keras.constraints.max_norm(self.max_norm))(convm)
            convm = batch_norm_activation(convm)
            convm = Conv2D(self.featuremaps * 16, (3, 3), padding=self.padding,
                           kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                           kernel_constraint=keras.constraints.max_norm(self.max_norm))(convm)
            convm = batch_norm_activation(convm)
            convm = layers.add([shortcut3, convm])

            deconv4 = Conv2DTranspose(self.featuremaps * 8, (2, 2), strides=(2, 2), padding=self.padding,
                                      kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                                      kernel_constraint=keras.constraints.max_norm(self.max_norm))(convm)
            uconv4 = concatenate([deconv4, conv4])
            uconv4 = Conv2D(self.featuremaps * 8, (2, 2), padding=self.padding,
                            kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                            kernel_constraint=keras.constraints.max_norm(self.max_norm))(uconv4)
            uconv4 = batch_norm_activation(uconv4)
            shortcut4 = Conv2D(self.featuremaps * 8, (3, 3), padding=self.padding,
                               kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                               kernel_constraint=keras.constraints.max_norm(self.max_norm))(uconv4)
            shortcut4 = batch_norm_activation(shortcut4)
            uconv4 = Conv2D(self.featuremaps * 8, (2, 2), padding=self.padding,
                            kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                            kernel_constraint=keras.constraints.max_norm(self.max_norm))(uconv4)
            uconv4 = batch_norm_activation(uconv4)
            uconv4 = Conv2D(self.featuremaps * 8, (2, 2), padding=self.padding,
                            kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                            kernel_constraint=keras.constraints.max_norm(self.max_norm))(uconv4)
            uconv4 = batch_norm_activation(uconv4)
            uconv4 = layers.add([shortcut4, uconv4])

            deconv3 = Conv2DTranspose(self.featuremaps * 4, (2, 2), strides=(2, 2), padding=self.padding,
                                      kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                                      kernel_constraint=keras.constraints.max_norm(self.max_norm))(uconv4)
            uconv3 = concatenate([deconv3, conv3])
            uconv3 = Conv2D(self.featuremaps * 4, (2, 2), padding=self.padding,
                            kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                            kernel_constraint=keras.constraints.max_norm(self.max_norm))(uconv3)
            uconv3 = batch_norm_activation(uconv3)
            shortcut5 = Conv2D(self.featuremaps * 4, (3, 3), padding=self.padding,
                               kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                               kernel_constraint=keras.constraints.max_norm(self.max_norm))(uconv3)
            shortcut5 = batch_norm_activation(shortcut5)
            uconv3 = Conv2D(self.featuremaps * 4, (2, 2), padding=self.padding,
                            kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                            kernel_constraint=keras.constraints.max_norm(self.max_norm))(uconv3)
            uconv3 = batch_norm_activation(uconv3)
            uconv3 = Conv2D(self.featuremaps * 4, (2, 2), padding=self.padding,
                            kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                            kernel_constraint=keras.constraints.max_norm(self.max_norm))(uconv3)
            uconv3 = batch_norm_activation(uconv3)
            uconv3 = layers.add([shortcut5, uconv3])

            deconv2 = Conv2DTranspose(self.featuremaps * 2, (2, 2), strides=(2, 2), padding=self.padding,
                                      kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                                      kernel_constraint=keras.constraints.max_norm(self.max_norm))(uconv3)
            uconv2 = concatenate([deconv2, conv2])
            uconv2 = Conv2D(self.featuremaps * 2, (2, 2), padding=self.padding,
                            kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                            kernel_constraint=keras.constraints.max_norm(self.max_norm))(uconv2)
            uconv2 = batch_norm_activation(uconv2)
            shortcut6 = Conv2D(self.featuremaps * 2, (3, 3), padding=self.padding,
                               kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                               kernel_constraint=keras.constraints.max_norm(self.max_norm))(uconv2)
            shortcut6 = batch_norm_activation(shortcut6)
            uconv2 = Conv2D(self.featuremaps * 2, (2, 2), padding=self.padding,
                            kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                            kernel_constraint=keras.constraints.max_norm(self.max_norm))(uconv2)
            uconv2 = batch_norm_activation(uconv2)
            uconv2 = Conv2D(self.featuremaps * 2, (2, 2), padding=self.padding,
                            kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                            kernel_constraint=keras.constraints.max_norm(self.max_norm))(uconv2)
            uconv2 = batch_norm_activation(uconv2)
            uconv2 = layers.add([shortcut6, uconv2])

            deconv1 = Conv2DTranspose(self.featuremaps * 1, (2, 2), strides=(2, 2), padding=self.padding,
                                      kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                                      kernel_constraint=keras.constraints.max_norm(self.max_norm))(uconv2)
            uconv1 = concatenate([deconv1, conv1])
            uconv1 = Conv2D(self.featuremaps * 1, (2, 2), padding=self.padding,
                            kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                            kernel_constraint=keras.constraints.max_norm(self.max_norm))(uconv1)
            uconv1 = batch_norm_activation(uconv1)
            shortcut7 = Conv2D(self.featuremaps * 1, (3, 3), padding=self.padding,
                               kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                               kernel_constraint=keras.constraints.max_norm(self.max_norm))(uconv1)
            shortcut7 = batch_norm_activation(shortcut7)
            uconv1 = Conv2D(self.featuremaps * 1, (2, 2), padding=self.padding,
                            kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                            kernel_constraint=keras.constraints.max_norm(self.max_norm))(uconv1)
            uconv1 = batch_norm_activation(uconv1)
            uconv1 = Conv2D(self.featuremaps * 1, (2, 2), padding=self.padding,
                            kernel_regularizer=keras.regularizers.L2(l2=self.L2),
                            kernel_constraint=keras.constraints.max_norm(self.max_norm))(uconv1)
            uconv1 = batch_norm_activation(uconv1)
            uconv1 = layers.add([shortcut7, uconv1])

            if self.num_class > 2:
                final = Conv2D(self.num_class, (1, 1), padding=self.padding, activation='softmax')(uconv1)
            elif self.num_class <= 2:
                final = Conv2D(1, (1, 1), padding=self.padding, activation='sigmoid')(uconv1)

            self.model = Model(inputs, final)

            if self.displaySummary:
                self.model.summary()

    def run(self):
        """Compiles and trains the model, with options for data augmentation and early stopping."""
        global steps_per_epoch, validation_steps
        callbacks = []

        if len(self.early_stopping) > 0:
            for early_stopping in self.early_stopping:
                early_stopping_tmp = keras.callbacks.EarlyStopping(monitor=early_stopping, mode='max', verbose=1,
                                                                   patience=self.patience, restore_best_weights=True)
                callbacks.append(early_stopping_tmp)
        if self.loss_early_stopping:
            early_stopping_loss = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                                                                patience=self.patience, restore_best_weights=True)
            callbacks.append(early_stopping_loss)

        print('gpu : ' + str(len(self.num_gpus)))
        if len(self.num_gpus) > 0:
            steps_per_epoch = self.x_train.shape[0] // (self.batch_size * len(self.num_gpus))
            validation_steps = self.x_test.shape[0] // (self.batch_size * len(self.num_gpus))
        elif len(self.num_gpus) == 0:
            steps_per_epoch = self.x_train.shape[0] // self.batch_size
            validation_steps = self.x_test.shape[0] // self.batch_size

        if self.data_generation:
            batch_size_val = self.batch_size // 2

            train_generator = self.data_generator(seed=105, batch_size=self.batch_size, is_train_set=True,
                                                  for_visualisation=False)
            test_generator = self.data_generator(seed=105, batch_size=batch_size_val, is_train_set=False,
                                                 for_visualisation=False)

            if self.additional_augmentation_factor != 0:
                steps_per_epoch = steps_per_epoch * self.additional_augmentation_factor

            self.model.compile(optimizer=Adam(self.learning_rate),
                               loss=self.loss,
                               metrics=self.metrics)

            self.model.fit_generator(generator=train_generator,
                                     validation_data=test_generator,
                                     steps_per_epoch=steps_per_epoch,
                                     validation_steps=validation_steps,
                                     # shuffle=True,
                                     epochs=self.epochs,
                                     callbacks=[callbacks])
        else:
            self.model.compile(optimizer=Adam(self.learning_rate),
                               loss=self.loss,
                               metrics=self.metrics)

            hist_model = self.model.fit(x=self.x_train,
                                        y=self.y_train,
                                        validation_data=(self.x_test, self.y_test),
                                        shuffle=True,
                                        epochs=self.epochs,
                                        batch_size=self.batch_size,
                                        steps_per_epoch=steps_per_epoch,
                                        validation_steps=validation_steps,
                                        callbacks=[callbacks])

            # Saves the history under the form of a plot
            training_history = hist_model.history
            metric_names = list(hist_model.history.keys())
            for metric_name in metric_names:
                plt.plot(training_history[metric_name], label=metric_name)
            plt.title('Metrics - ' + self.file_name)
            plt.xlabel('Epoch')
            plt.ylabel('Metric value')
            plt.legend(metric_names, loc='best')

            # Saves the model and the weights for future loading and usage
            if self.save_model:
                if not os.path.exists(self.logdir):
                    os.makedirs(self.logdir)
                # saves the plot
                plt.savefig(os.path.join(self.logdir, 'metrics.png'))
                # Save the scaler to a binary file
                if self.util is not None:
                    dump(self.util.scaler, os.path.join(self.logdir, 'std_scaler_image.bin'), compress=True)
                    # Saves the log file containing the variables used to compile the model
                with open(self.logdir + '/logs.txt', 'w') as f:
                    f.write(self.FILE_TXT)
                model_json = self.model.to_json()
                new_path = os.path.join(self.logdir, 'weights')
                print(new_path)
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                with open(os.path.join(new_path, 'weights.json'), "w") as json_file:
                    json_file.write(model_json)
                    self.model.save_weights(os.path.join(new_path, 'weights.h5'))
                    self.model.save(new_path)
