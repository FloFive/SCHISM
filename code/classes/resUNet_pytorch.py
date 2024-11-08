import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from util import Util


class ResUNet:
    def __repr__(self):
        """Returns a string representation of the class."""
        return 'ResUNet'

    def __init__(self, **kwargs):
        """Initializes the ResUNet class with given parameters."""
        self.model = None
        self.validation_split = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                self.x_train = self.util.get_x_train()
                self.y_train = self.util.get_y_train()
                self.x_test = self.util.get_x_test()
                self.y_test = self.util.get_y_test()
                self.num_class = self.util.get_num_class()
                self.FILE_TXT = self.FILE_TXT + "\nnumClass = " + str(self.num_class)
                self.FILE_TXT = self.FILE_TXT + "\nnumSample = " + str(self.util.get_num_slice())
                self.FILE_TXT = self.FILE_TXT + "\nval_split = " + str(self.util.get_validation_split())
                self.FILE_TXT = self.FILE_TXT + "\nimage_preprocessing_functions = ["
                if len(self.util.get_image_preprocessing_functions()) > 0:
                    marker = len(self.util.get_image_preprocessing_functions()) - 1
                    i = 0
                    for function in self.util.get_image_preprocessing_functions():
                        if len(self.util.get_image_preprocessing_functions()) == 1:
                            self.FILE_TXT = self.FILE_TXT + function
                        elif len(self.util.get_image_preprocessing_functions()) > 1:
                            if marker == i:
                                self.FILE_TXT = self.FILE_TXT + function
                            else:
                                self.FILE_TXT = self.FILE_TXT + function + ","
                        i += 1
                self.FILE_TXT = self.FILE_TXT + "]"
                self.FILE_TXT = self.FILE_TXT + "\nimageType = " + str(self.util.getImageType())
            else:
                raise Exception(repr(self) + ' class - training and testing dataset must be provided')

        if "epochs" in kwargs:  # Total number of iterations over the entire training dataset
            self.epochs = int(kwargs['epochs'])
        else:
            self.epochs = 50
        self.FILE_TXT = self.FILE_TXT + "\nepochs = " + str(self.epochs)

        if "batch_size" in kwargs:  # Number of samples per gradient update
            self.batch_size = int(kwargs['batch_size'])
        else:
            self.batch_size = 8
        self.FILE_TXT = self.FILE_TXT + "\nbatch_size = " + str(self.batch_size)

        if "learningRate" in kwargs:  # Step size for updating the weights
            self.learning_rate = float(kwargs['learningRate'])
        else:
            self.learning_rate = 1e-5
        self.FILE_TXT = self.FILE_TXT + "\nlearningRate = " + str(self.learning_rate)

        if "L2" in kwargs:  # L2 regularization factor to prevent over-fitting
            self.L2 = float(kwargs['L2'])
        else:
            self.L2 = 1e-5
        self.FILE_TXT = self.FILE_TXT + "\nL2 = " + str(self.L2)

        if "batchNorm" in kwargs:  # Whether to use batch normalization
            self.batch_norm = bool(kwargs['batchNorm'])
            self.FILE_TXT = self.FILE_TXT + "\nbatchNorm = " + str(self.batch_norm)
        else:
            self.batch_norm = False
            self.FILE_TXT = self.FILE_TXT + "\nbatchNorm = none"

        if "maxNorm" in kwargs:  # Maximum norm constraint for weights
            self.max_norm = int(kwargs['maxNorm'])
        else:
            self.max_norm = 1
        self.FILE_TXT = self.FILE_TXT + "\nMaxNorm = " + str(self.max_norm)

        if "dropOut" in kwargs:  # Whether to use dropout to prevent over-fitting
            self.drop_out = bool(kwargs['dropOut'])
            self.FILE_TXT = self.FILE_TXT + "\ndropOut = " + str(self.drop_out)
        else:
            self.drop_out = False
            self.FILE_TXT = self.FILE_TXT + "\ndropOut = none"

        if self.drop_out:
            if "dropoutRate" in kwargs:  # Dropout rate for regularization
                self.dropout_rate = float(kwargs['dropoutRate'])
            else:
                self.dropout_rate = 0.1
            self.FILE_TXT = self.FILE_TXT + "\ndropoutRate = " + str(self.dropout_rate)
        else:
            self.dropout_rate = None
            self.FILE_TXT = self.FILE_TXT + "\ndropoutRate = none"

        if "dataGeneration" in kwargs:  # Whether to use data augmentation
            self.data_generation = bool(kwargs['dataGeneration'])
        else:
            self.data_generation = False
        self.FILE_TXT = self.FILE_TXT + "\ndataGeneration = " + str(self.data_generation)

        if "additional_augmentation_factor" in kwargs:  # Factor for additional data augmentation
            self.additional_augmentation_factor = kwargs['additional_augmentation_factor']
        else:
            self.additional_augmentation_factor = 2
        self.FILE_TXT = self.FILE_TXT + "\nadditional_augmentation_factor = " + str(
            self.additional_augmentation_factor)

        if "patience" in kwargs:  # Number of epochs with no improvement after which training will be stopped
            self.patience = int(kwargs['patience'])
        else:
            self.patience = 5
        self.FILE_TXT = self.FILE_TXT + "\npatience = " + str(self.patience)

        if "padding" in kwargs:  # Padding strategy for convolutional layers
            if kwargs['padding'] == "same" or kwargs['padding'] == "valid":
                self.padding = kwargs['padding']
            else:
                self.padding = "same"
        else:
            self.padding = "same"
        self.FILE_TXT = self.FILE_TXT + "\npadding = " + str(self.padding)

        if "pretrained" in kwargs:  # Whether to use a pretrained model
            self.pretrained = bool(kwargs['pretrained'])
            if "backbone" in kwargs:  # Backbone model to use if pretrained
                self.backbone = str(kwargs['backbone'])
                self.FILE_TXT = self.FILE_TXT + "\nbackbone= " + str(self.backbone)
        else:
            self.pretrained = False
        self.FILE_TXT = self.FILE_TXT + "\npretrained = " + str(self.pretrained)

        if "img_height" in kwargs:  # Height of the input images
            self.img_height = int(kwargs['img_height'])
        else:
            self.img_height = self.x_train.shape[1]
        self.FILE_TXT = self.FILE_TXT + "\nimg_height = " + str(self.img_height)

        if "img_width" in kwargs:  # Width of the input images
            self.img_width = int(kwargs['img_width'])
        else:
            self.img_width = self.x_train.shape[2]
        self.FILE_TXT = self.FILE_TXT + "\nimg_width = " + str(self.img_width)

        metrics_early_stopping = []
        metrics_early_stopping_tmp = []
        if "metrics" in kwargs:  # Metrics to monitor during training
            if "early_stopping" in kwargs:  # Metrics for early stopping
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
            else:  # Multiclass
                self.loss = self.weighted_categorical_crossentropy()

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
        return [f'cuda:{i}' for i in range(torch.cuda.device_count())]

    def get_validation_split(self):
        """Returns the validation split."""
        return self.validation_split

    def data_generator(self, seed, batch_size, is_train_set, for_visualisation=False):
        """
        Generates batches of data for training or testing.
        """
        np.random.seed(seed)

        if is_train_set:
            transform = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor()
            ])
        else:
            transform = transforms.ToTensor()

        train_dataset = self.CustomDataset(self.x_train, self.y_train, transform=transform)
        test_dataset = self.CustomDataset(self.x_test, self.y_test, transform=transforms.ToTensor())

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=is_train_set)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        while True:
            for data in train_loader if is_train_set else test_loader:
                yield data

    def get_model(self):
        """Returns the current model."""
        return self.model

    def weighted_categorical_crossentropy(self):
        """Returns a weighted categorical crossentropy loss function."""
        weights = torch.tensor(self.util.getClassFrequency(), dtype=torch.float32).to(self.device)

        def loss(y_true, y_pred):
            y_pred = torch.clamp(y_pred, min=1e-7, max=1 - 1e-7)  # Clip to prevent NaN's and Inf's
            loss_in = -y_true * torch.log(y_pred) * weights
            return loss_in.mean()

        return loss

    def weighted_binary_crossentropy(self):
        """Returns a weighted binary crossentropy loss function."""

        def loss(y_true, y_pred):
            bin_crossentropy = nn.BCELoss(reduction='none')(y_pred, y_true)
            weights = y_true * self.util.getClassFrequency()[1] + (1. - y_true) * self.util.getClassFrequency()[0]
            weighted_bin_crossentropy = weights * bin_crossentropy
            return weighted_bin_crossentropy.mean()

        return loss

    def set_model(self, **kwargs):
        """Configures and sets up the U-Net model architecture."""
        self.featuremaps = kwargs.get('featuremaps', 8)
        self.padding = kwargs.get('padding', 'same')
        self.L2 = kwargs.get('L2', 1e-5)
        self.max_norm = kwargs.get('maxNorm', 1)
        self.batch_norm = kwargs.get('batchNorm', False)
        self.drop_out = kwargs.get('dropOut', False)
        self.dropout_rate = kwargs.get('dropoutRate', 0.1) if self.drop_out else None

        # Define the U-Net architecture
        self.model = nn.Module()
        self.model.encoder1 = self.conv_block(3, self.featuremaps)
        self.model.encoder2 = self.conv_block(self.featuremaps, self.featuremaps * 2)
        self.model.encoder3 = self.conv_block(self.featuremaps * 2, self.featuremaps * 4)
        self.model.encoder4 = self.conv_block(self.featuremaps * 4, self.featuremaps * 8)
        self.model.bottleneck = self.conv_block(self.featuremaps * 8, self.featuremaps * 16)

        self.model.decoder4 = self.upconv_block(self.featuremaps * 16, self.featuremaps * 8)
        self.model.decoder3 = self.upconv_block(self.featuremaps * 8, self.featuremaps * 4)
        self.model.decoder2 = self.upconv_block(self.featuremaps * 4, self.featuremaps * 2)
        self.model.decoder1 = self.upconv_block(self.featuremaps * 2, self.featuremaps)

        self.model.final_conv = nn.Conv2d(self.featuremaps, self.num_class, kernel_size=1)

        self.model.to(self.device)

    def conv_block(self, in_channels, out_channels):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if self.batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if self.batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def upconv_block(self, in_channels, out_channels):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            self.conv_block(out_channels, out_channels)
        ]
        return nn.Sequential(*layers)

    def run(self):
        """Compiles and trains the model, with options for data augmentation and early stopping."""
        steps_per_epoch = len(self.x_train) // self.batch_size
        validation_steps = len(self.x_test) // self.batch_size

        # Early stopping setup
        # early_stopping = None
        # if self.early_stopping:
        #   early_stopping = EarlyStopping(patience=self.patience, verbose=1)

        # Data loaders
        if self.data_generation:
            batch_size_val = self.batch_size // 2

            train_loader = self.data_generator(seed=105, batch_size=self.batch_size, is_train_set=True)
            test_loader = self.data_generator(seed=105, batch_size=batch_size_val, is_train_set=False)

            if self.additional_augmentation_factor != 0:
                steps_per_epoch *= self.additional_augmentation_factor

        else:
            train_dataset = self.CustomDataset(self.x_train, self.y_train, transform=transforms.ToTensor())
            test_dataset = self.CustomDataset(self.x_test, self.y_test, transform=transforms.ToTensor())
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = self.loss  # Assuming self.loss is already defined as a PyTorch loss function

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {running_loss / steps_per_epoch: .4f}')

            # Validation step
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            print(f'Validation Loss: {val_loss / validation_steps: .4f}')

            # Early stopping check
            # if early_stopping:
            #     early_stopping(val_loss / validation_steps, self.model)
            #     if early_stopping.early_stop:
            #         print("Early stopping")
            #         break

        # Save metrics plot
        # // Error as epoch and running_loss are declared in the for context (line 416) but used outside //
        plt.plot(range(1, epoch + 2), running_loss, label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.logdir, 'metrics.png'))

        # Save the model and weights
        if self.save_model:
            if not os.path.exists(self.logdir):
                os.makedirs(self.logdir)
            torch.save(self.model.state_dict(), os.path.join(self.logdir, 'model.pth'))

            # Save logs
            with open(os.path.join(self.logdir, 'logs.txt'), 'w') as f:
                f.write(self.FILE_TXT)

    class CustomDataset(Dataset):
        def __init__(self, images, labels, transform=None):
            self.images = images
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = self.images[idx]
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, label

    # Example usage
    # res_unet = ResUNet(pathLogDir='logs', data=Util(), epochs=50, batch_size=8)
    # res_unet.run()
