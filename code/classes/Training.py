import sys
import os
import torch
import torch.nn as nn
import numpy as np
import glob
import json
import matplotlib
matplotlib.use('Agg') 
from tqdm import tqdm
from classes.TiffDatasetLoader import TiffDatasetLoader
from classes.ParamConverter import ParamConverter
from classes.TrainingLogger import TrainingLogger
from classes.model_registry import model_mapping
from datetime import datetime
from torch.optim import Adagrad, Adam, AdamW, NAdam, RMSprop, RAdam, SGD
from torch.optim.lr_scheduler import LRScheduler, LambdaLR, MultiplicativeLR, StepLR, MultiStepLR, ConstantLR, LinearLR, ExponentialLR, PolynomialLR, CosineAnnealingLR, SequentialLR, ReduceLROnPlateau, CyclicLR, OneCycleLR, CosineAnnealingWarmRestarts
from torchmetrics.classification import BinaryJaccardIndex, MulticlassJaccardIndex, MulticlassF1Score, BinaryF1Score, BinaryAccuracy, MulticlassAccuracy, BinaryAveragePrecision, MulticlassAveragePrecision, BinaryConfusionMatrix, MulticlassConfusionMatrix, BinaryPrecision, MulticlassPrecision, BinaryRecall, MulticlassRecall
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, NLLLoss
from torch.utils.data import DataLoader
import torch.nn.functional as nn_func
import torch.backends.cudnn as cudnn
from early_stopping_pytorch import EarlyStopping

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Training:

    def __repr__(self):
        """
        Returns a string representation of the Training class.

        Returns:
            str: A string indicating the class name.
        """
        return 'Training'

    def __init__(self, **kwargs):
        """
        Initializes the Training class with given parameters.

        Args:
            **kwargs: Keyword arguments containing configuration parameters such as:
                - subfolders (list): List of subfolder names containing the dataset.
                - data_dir (str): Directory containing the dataset.
                - run_dir (str): Directory for saving model outputs.
                - hyperparameters (Hyperparameters): An object containing model and training parameters.

        Raises:
            Exception: If pathLogDir is not provided.
        """
        self.optimizer_mapping = {
            'Adagrad' : Adagrad, 
            'Adam' : Adam, 
            'AdamW' : AdamW, 
            'NAdam' : NAdam, 
            'RMSprop' : RMSprop, 
            'RAdam' : RAdam, 
            'SGD' : SGD
        }

        self.loss_mapping = {
            'CrossEntropyLoss' : CrossEntropyLoss, 
            'BCEWithLogitsLoss' : BCEWithLogitsLoss, 
            'NLLLoss' : NLLLoss, 
        }

        self.scheduler_mapping = {
            'LRScheduler': LRScheduler,
            'LambdaLR': LambdaLR,
            'MultiplicativeLR': MultiplicativeLR,
            'StepLR': StepLR,
            'MultiStepLR': MultiStepLR,
            'ConstantLR': ConstantLR,
            'LinearLR': LinearLR,
            'ExponentialLR': ExponentialLR,
            'PolynomialLR': PolynomialLR,
            'CosineAnnealingLR': CosineAnnealingLR,
            'SequentialLR': SequentialLR,
            'ReduceLROnPlateau': ReduceLROnPlateau,
            'CyclicLR': CyclicLR,
            'OneCycleLR': OneCycleLR,
            'CosineAnnealingWarmRestarts': CosineAnnealingWarmRestarts
        }
        
        self.param_converter = ParamConverter()  
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.subfolders = kwargs.get('subfolders')
        self.data_dir = kwargs.get('data_dir')
        self.run_dir = kwargs.get('run_dir')
        self.hyperparameters = kwargs.get('hyperparameters')
       
        #Model parameters
        self.model_params = {k: v for k, v in self.hyperparameters.get_parameters()['Model'].items()}
        self.num_classes = self.param_converter._convert_param(self.model_params.get('num_classes', 3))
        self.num_classes = 1 if self.num_classes <= 2 else self.num_classes
        self.model_mapping = model_mapping
        self.model = self.initialize_model()
       
        #Optimizer parameters
        self.optimizer_params = {k: v for k, v in self.hyperparameters.get_parameters()['Optimizer'].items()}
        self.optimizer = self.initialize_optimizer()
        
        #Scheduler parameters
        self.scheduler_params = {k: v for k, v in self.hyperparameters.get_parameters()['Scheduler'].items()}
        self.scheduler = self.initialize_scheduler(optimizer=self.optimizer)

        #Loss parameters
        self.loss_params = {k: v for k, v in self.hyperparameters.get_parameters()['Loss'].items()}
        self.weights = self.param_converter._convert_param(self.loss_params.get('weights', "False"))
        self.ignore_background = self.param_converter._convert_param(self.loss_params.get('ignore_background', "False"))
            
        # Training parameters
        self.training_params = {k: v for k, v in self.hyperparameters.get_parameters()['Training'].items()}
        self.batch_size = self.param_converter._convert_param(self.training_params.get('batch_size', 8))
        self.val_split = self.param_converter._convert_param(self.training_params.get('val_split', 0.8))
        self.epochs = self.param_converter._convert_param(self.training_params.get('epochs', 10))
        self.early_stopping = self.param_converter._convert_param(self.training_params.get('early_stopping', "False"))
        self.metrics_str = self.param_converter._convert_param(self.training_params.get('metrics', ''))        
       
        # Data parameters
        self.data = {k: v for k, v in self.hyperparameters.get_parameters()['Data'].items()}
        self.img_res = self.param_converter._convert_param(self.data.get('img_res', 560))
        self.crop_size = self.param_converter._convert_param(self.data.get('crop_size', 224))
        self.num_samples = self.param_converter._convert_param(self.data.get('num_samples', 500))

        self.training_time = datetime.now().strftime("%d-%m-%y-%H-%M-%S")

        if self.ignore_background:
            self.ignore_index = -1 
        else:
            self.ignore_index = -100
            
        if self.early_stopping:
            patience = int(self.epochs*0.2)
            if patience > 1:
                self.early_stopping_instance = EarlyStopping(patience=patience, verbose=True)
            else:
                self.early_stopping=False
                print("Early stopping has been automatically disabled because the patience value is too low.")
                print("Training will begin as normal.")

        self.save_directory = self.create_unique_folder()
        self.logger = TrainingLogger(save_directory=self.save_directory,
                                    num_classes=self.num_classes,
                                    model_params=self.model_params,
                                    optimizer_params=self.optimizer_params,
                                    scheduler_params=self.scheduler_params,
                                    loss_params=self.loss_params,
                                    training_params=self.training_params,
                                    data=self.data)
                                    
    def create_metric(self, binary_metric, multiclass_metric):
        return (
            binary_metric(ignore_index=self.ignore_index).to(self.device)
            if self.num_classes == 1
            else multiclass_metric(num_classes=self.num_classes, ignore_index=self.ignore_index).to(self.device)
        )

    def initialize_metrics(self):
        """
        Initializes the specified metrics for evaluation.
        
        Returns:
            list: A list of metric instances corresponding to the specified names.
        
        Raises:
            ValueError: If a specified metric is not recognized.
        """
        self.metrics_mapping = {
            "Jaccard": self.create_metric(BinaryJaccardIndex, MulticlassJaccardIndex),
            "F1": self.create_metric(BinaryF1Score, MulticlassF1Score),
            "Accuracy": self.create_metric(BinaryAccuracy, MulticlassAccuracy),
            "AveragePrecision": self.create_metric(BinaryAveragePrecision, MulticlassAveragePrecision),
            "ConfusionMatrix": self.create_metric(BinaryConfusionMatrix, MulticlassConfusionMatrix),
            "Precision": self.create_metric(BinaryPrecision, MulticlassPrecision),
            "Recall": self.create_metric(BinaryRecall, MulticlassRecall),
        }
    
        # Parse metrics from string input or use default
        self.metrics = [metric.strip() for metric in self.metrics_str.split(',')] if self.metrics_str else ["Jaccard"]
    
        # Retrieve metric instances
        selected_metrics = []
        for metric in self.metrics:
            if metric in self.metrics_mapping:
                selected_metrics.append(self.metrics_mapping[metric])
            else:
                raise ValueError(f"Metric '{metric}' not recognized. Please check the name.")
    
        return selected_metrics

    def initialize_optimizer(self):
        optimizer_name = self.optimizer_params.get('optimizer', 'Adam')
        optimizer_class = self.optimizer_mapping.get(optimizer_name)

        if not optimizer_class:
            raise ValueError(f"Optimizer '{optimizer_name}' is not supported. Check your 'optimizer_mapping'.")

        converted_params = {k: self.param_converter._convert_param(v) for k, v in self.optimizer_params.items() if k != 'optimizer'}

        return optimizer_class(self.model.parameters(), **converted_params)

    def initialize_scheduler(self, optimizer):
        scheduler_name = self.scheduler_params.get('scheduler', 'ConstantLR')
        scheduler_class = self.scheduler_mapping.get(scheduler_name)

        if not scheduler_class:
            raise ValueError(f"Scheduler '{scheduler_name}' is not supported. Check your 'scheduler_mapping'.")

        converted_params = {k: self.param_converter._convert_param(v) for k, v in self.scheduler_params.items() if k != 'scheduler'}

        if not converted_params:
            return scheduler_class(optimizer)
        else:
            return scheduler_class(optimizer, **converted_params)

    def initialize_loss(self, **dynamic_params):
        loss_name = self.loss_params.get('loss', 'CrossEntropyLoss')
        loss_class = self.loss_mapping.get(loss_name)

        if not loss_class:
            raise ValueError(f"Loss '{loss_name}' is not supported. Check your 'loss_mapping'.")

        # Convert static parameters from config
        converted_params = {
            k: self.param_converter._convert_param(v) 
            for k, v in self.loss_params.items() 
            if k not in {'loss', 'ignore_background', 'weights'}  # Exclude unwanted params
        }

        # Merge with dynamic parameters (e.g., batch-specific weights)
        final_params = {**converted_params, **dynamic_params}

        # Check if ignore_index should be included (for all losses except BCEWithLogitsLoss)
        if loss_name == 'BCEWithLogitsLoss':
            final_params.pop('ignore_index', None)  # Remove ignore_index if not needed
        else:
            if self.num_classes > 1:
                final_params['ignore_index'] = self.ignore_index
            else:
                final_params.pop('ignore_index', None)  # Remove ignore_index if not needed

        return loss_class(**final_params)

    def initialize_model(self) -> nn.Module:
        model_name = self.model_params.get('model_type', 'UnetVanilla')

        if model_name not in self.model_mapping:
            raise ValueError(f"Model '{model_name}' is not supported. Check your 'model_mapping'.")

        model_class = self.model_mapping[model_name]
        self.model_params['num_classes'] = self.num_classes

        required_params = {
            k: self.param_converter._convert_param(v) for k, v in self.model_params.items() if k in model_class.REQUIRED_PARAMS
        }
        optional_params = {
            k: self.param_converter._convert_param(v) for k, v in self.model_params.items() if k in model_class.OPTIONAL_PARAMS
        }

        required_params.pop('model_type', None)
        optional_params.pop('model_type', None)

        try:
            typed_required_params = {
                k: model_class.REQUIRED_PARAMS[k](v) 
                for k, v in required_params.items()
            }

            typed_optional_params = {
                k: model_class.OPTIONAL_PARAMS[k](v) 
                for k, v in optional_params.items()
            }
        except ValueError as e:
            raise ValueError(f"Error converting parameters for model '{model_name}': {e}")

        return model_class(**typed_required_params, **typed_optional_params).to(self.device)

    def create_unique_folder(self):
        """
        Creates a unique folder for saving model weights and logs based on the current training parameters.

        Returns:
            str: The path to the created directory.
        """
        filename = f"{self.model_params.get('model_type', 'UnetVanilla')}__" \
            f"{self.training_time}"

        save_directory = os.path.join(self.run_dir, filename)

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        return save_directory

    def load_segmentation_data(self):
        """
        Loads segmentation data from the specified directories, prepares datasets, and creates data loaders.

        This method also handles the loading of normalization statistics and the splitting of data into training,
        validation, and test sets.
        """

        def load_data_stats(data_dir):
            """
            Loads normalization statistics from a JSON file. Provides default normalization
            stats if the file is missing or improperly formatted.

            Args:
                data_dir (str): Directory containing the data stats JSON file.

            Returns:
                dict: A dictionary containing the loaded data statistics.
            """
            neutral_stats = [np.array([0.5] * 3), np.array([0.5] * 3)]  # Default mean and std
            json_file_path = os.path.join(data_dir, 'data_stats.json')

            if not os.path.exists(json_file_path):
                print(f"File {json_file_path} not found. Using default normalization stats.")
                return {"default": neutral_stats}

            try:
                with open(json_file_path, 'r') as file:
                    raw_data_stats = json.load(file)

                data_stats_loaded = {}
                for key, value in raw_data_stats.items():
                    if not (isinstance(value, list) and len(value) == 2 and
                            all(isinstance(v, list) and len(v) == 3 for v in value)):
                        raise ValueError(f"Invalid format in data_stats.json for key {key}")

                    data_stats_loaded[key] = [
                        np.array(value[0]),
                        np.array(value[1])
                    ]

                return data_stats_loaded

            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error loading data stats from {json_file_path}: {e}. Using default normalization stats.")
                return {"default": neutral_stats}

        def generate_random_indices(num_samples, val_split, subfolders, num_sample_subfolder):
            """
            Generates random indices for splitting the dataset into training, validation, and test sets.

            Args:
                num_samples (int): Total number of samples in the dataset.
                val_split (float): Proportion of the dataset to use for validation.
                subfolders (list): List of subfolder names containing the dataset.
                num_sample_subfolder (dict): Dictionary mapping subfolder names to the number of samples in each.

            Returns:
                tuple: Three lists containing the indices for training, validation, and test sets.
            """
            num_train = int(num_samples * val_split)
            num_val = num_samples - num_train
            all_indices = [
                (sub_folder_temp, sample_id)
                for sub_folder_temp in subfolders
                for sample_id in range(num_sample_subfolder[sub_folder_temp])
            ]
            np.random.shuffle(all_indices)
            val_indices_temp = np.random.choice(len(all_indices), size=num_val, replace=False)
            val_set = set(val_indices_temp)
            train_indices_temp = [
                                     idx for idx in range(len(all_indices)) if idx not in val_set
                                 ][:num_train]
            train_set = set(train_indices_temp)
            all_indices_set = set(range(len(all_indices)))
            test_indices_temp = list(all_indices_set - train_set - val_set)
            train_indices_temp = [all_indices[i] for i in train_indices_temp]
            val_indices_temp = [all_indices[i] for i in val_indices_temp]
            test_indices_temp = [all_indices[i] for i in test_indices_temp]
            return train_indices_temp, val_indices_temp, test_indices_temp

        img_data = {}
        mask_data = {}
        num_sample_per_subfolder = {}
        data_stats = load_data_stats(self.data_dir)

        for subfolder in self.subfolders:
            img_folder = os.path.join(self.data_dir, subfolder, "images")
            mask_folder = os.path.join(self.data_dir, subfolder, "masks")
            img_files = sorted(glob.glob(os.path.join(img_folder, "*")))
            mask_files = sorted(glob.glob(os.path.join(mask_folder, "*")))
            assert len(img_files) == len(mask_files), (
                f"Mismatch: {len(img_files)} images, {len(mask_files)} masks in {subfolder}"
            )
            img_data[subfolder] = img_files
            mask_data[subfolder] = mask_files
            num_sample_per_subfolder[subfolder] = len(img_files)

        train_indices, val_indices, test_indices = generate_random_indices(
            num_samples=self.num_samples,
            val_split=self.val_split,
            subfolders=self.subfolders,
            num_sample_subfolder=num_sample_per_subfolder,
        )
        
        indices = [train_indices, val_indices, test_indices]

        train_dataset = TiffDatasetLoader(
            indices=train_indices,
            img_data=img_data,
            mask_data=mask_data,
            num_classes=self.num_classes,
            crop_size=(self.crop_size, self.crop_size),
            data_stats=data_stats,
            img_res=self.img_res, 
            ignore_background=self.ignore_background,
            weights=self.weights
        )
        val_dataset = TiffDatasetLoader(
            indices=val_indices,
            img_data=img_data,
            mask_data=mask_data,
            num_classes=self.num_classes,
            crop_size=(self.crop_size, self.crop_size),
            data_stats=data_stats,
            img_res=self.img_res,
            ignore_background=self.ignore_background,
            weights=self.weights
        )
        test_dataset = TiffDatasetLoader(
            indices=test_indices,
            img_data=img_data,
            mask_data=mask_data,
            num_classes=self.num_classes,
            crop_size=(self.crop_size, self.crop_size),
            data_stats=data_stats,
            img_res=self.img_res,
            ignore_background=self.ignore_background,
            weights=self.weights
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, 
                                 pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=2, drop_last=True)

        self.logger.save_indices_to_file([train_indices, val_indices, test_indices])

        self.dataloaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'indices': indices,
        }

    def training_loop(self, optimizer, scheduler):
        
        def print_epoch_box(epoch, total_epochs):
            epoch_str = f" Epoch {epoch}/{total_epochs} "
            box_width = len(epoch_str) + 4
            print(f"╔{'═' * (box_width - 2)}╗")
            print(f"║{epoch_str.center(box_width - 2)}║")
            print(f"╚{'═' * (box_width - 2)}╝")
        
        scaler = None
        if self.device == "cuda":
            scaler = torch.amp.GradScaler()
            torch.backends.cudnn.benchmark = True  # Optimize conv layers

        metrics = self.initialize_metrics()
        loss_dict = {"train": {}, "val": {}}
        display_metrics = [m for m in self.metrics if m != "ConfusionMatrix"]
        metrics_dict = {phase: {metric: [] for metric in display_metrics} for phase in ["train", "val"]}
        best_val_loss = float("inf")
        best_val_metrics = {metric: 0 for metric in display_metrics}

        for epoch in range(1, self.epochs + 1):
            print_epoch_box(epoch, self.epochs)
            epoch_val_loss = None if self.early_stopping else None

            for phase in ["train", "val"]:
                is_training = (phase == "train")
                self.model.train() if is_training else self.model.eval()

                running_loss = 0.0
                running_metrics = {metric: 0.0 for metric in display_metrics}
                total_samples = 0

                with tqdm(total=len(self.dataloaders[phase]), unit="batch", leave=True) as pbar:
                    for inputs, labels, weights in self.dataloaders[phase]:
                        # Move data to the proper device
                        inputs, labels, weights = inputs.to(self.device), labels.to(self.device), weights.to(self.device)

                        optimizer.zero_grad()
                        batch_weights = torch.mean(weights, dim=0)
                        batch_weights = torch.clamp(batch_weights, min=1e-6)  # Avoid exact zero values
                        with torch.set_grad_enabled(is_training):
                          
                            # Forward pass under autocast using bfloat16
                            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                                outputs = self.model(inputs)

                            # If using NLLLoss, apply log_softmax to outputs
                            if self.loss_params.get('loss') in ['NLLLoss']:
                                outputs = torch.nn.functional.log_softmax(outputs, dim=1)

                            # Adjust label type based on number of classes:
                            if self.num_classes == 1:
                                outputs = outputs.squeeze()
                                # For binary segmentation using BCE, targets must be float
                                labels = labels.squeeze().float()
                            else:
                                # For multi-class segmentation with CrossEntropyLoss, targets must be long
                                labels = labels.squeeze().long()

                            # Initialize loss function (apply class weights only if applicable)
                            loss_fn = self.initialize_loss(weight=batch_weights if (self.weights and self.num_classes > 1) else None)

                            # Cast outputs to float32 before computing the loss
                            loss = loss_fn(outputs.float(), labels)
                           
                            if is_training:
                                if scaler:
                                    scaler.scale(loss).backward()
                                    scaler.step(optimizer)
                                    scaler.update()
                                else:
                                    loss.backward()
                                    optimizer.step()

                                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                    scheduler.step(loss)
                                else:
                                    scheduler.step()

                        running_loss += loss.item()
                        total_samples += labels.size(0)

                        with torch.no_grad():
                            preds = (torch.argmax(outputs, dim=1).long()
                                    if self.num_classes > 1
                                    else (outputs > 0.5).to(torch.uint8))
                            # Ensure labels are long for metric computations (for multi-class)
                            labels = labels.long()

                            for metric_name, metric_fn in zip(self.metrics, metrics):
                                if metric_name != "ConfusionMatrix":
                                    running_metrics[metric_name] += metric_fn(preds, labels).item()

                        pbar.set_postfix(
                            loss=running_loss / (pbar.n + 1),
                            **{metric: running_metrics[metric] / (pbar.n + 1) for metric in display_metrics}
                        )
                        pbar.update(1)

                epoch_loss = running_loss / len(self.dataloaders[phase])
                epoch_metrics = {metric: running_metrics[metric] / len(self.dataloaders[phase])
                                for metric in display_metrics}
                loss_dict[phase][epoch] = epoch_loss

                for metric, value in epoch_metrics.items():
                    metrics_dict[phase][metric].append(value)

                print(f"{phase.title()} Loss: {epoch_loss: .4f}")
                for metric, value in epoch_metrics.items():
                    print(f"{phase.title()} {metric}: {value: .4f}", end=" | ")
                print()

                if phase == "val" and epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    torch.save(self.model.state_dict(), os.path.join(self.save_directory, "model_best_loss.pth"))

                if phase == "val":
                    if self.early_stopping:
                        epoch_val_loss = epoch_loss

                    for metric, value in epoch_metrics.items():
                        if value > best_val_metrics[metric]:
                            best_val_metrics[metric] = value
                            torch.save(
                                self.model.state_dict(),
                                os.path.join(self.save_directory, f"model_best_{metric}.pth")
                            )

            if self.early_stopping and epoch_val_loss is not None:
                self.early_stopping_instance(epoch_val_loss, self.model)
                if self.early_stopping_instance.early_stop:
                    print("Early stopping triggered")
                    break

        print(f"Best Validation Metrics: {best_val_metrics}")
        
        return loss_dict, metrics_dict, metrics

    def train(self):
            loss_dict, metrics_dict, metrics = self.training_loop(optimizer=self.optimizer, 
                                                                scheduler=self.scheduler)
            
            #plot and metric saving
            self.logger.save_best_metrics(loss_dict=loss_dict, 
                                        metrics_dict=metrics_dict)
            self.logger.plot_learning_curves(loss_dict=loss_dict, 
                                            metrics_dict=metrics_dict)
            self.logger.save_hyperparameters()
            self.logger.save_data_stats(self.dataloaders["train"].dataset.data_stats)
            if "ConfusionMatrix" in self.metrics:
                self.logger.save_confusion_matrix(conf_metric=metrics[self.metrics.index("ConfusionMatrix")], 
                                                model=self.model, 
                                                val_dataloader=self.dataloaders["val"], 
                                                device=self.device)
