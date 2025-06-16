import os
import json
import configparser
import numpy as np
import torch
import matplotlib.pyplot as plt


class TrainingLogger:
    def __init__(self, save_directory, num_classes, model_params, optimizer_params, scheduler_params, loss_params, training_params, data, augmentation_mapping=False):
        """
        Initializes the TrainingLogger.

        Args:
            save_directory (str): Path to save logs and plots.
            num_classes (int): Number of classes for classification/segmentation.
            model_params (dict): Dictionary of model hyperparameters.
            optimizer_params (dict): Dictionary of optimizer hyperparameters.
            scheduler_params (dict): Dictionary of scheduler settings.
            training_params (dict): Dictionary of training hyperparameters.
            data (dict): Data-related parameters.
        """
        self.save_directory = save_directory
        self.num_classes = num_classes
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        self.loss_params = loss_params
        self.training_params = training_params
        self.data = data
        if augmentation_mapping:
            self.augmentation_mapping = augmentation_mapping

        os.makedirs(self.save_directory, exist_ok=True)
        
    def save_indices_to_file(self, indices_list):
        """
        Saves the indices of the training, validation, and test sets to text files.

        Args:
            indices_list (list): A list containing the indices for training, validation, and test sets.
        """
        indices_map = {
            "train": indices_list[0],
            "val": indices_list[1],
            "test": indices_list[2],
        }
        for idx_type, idx_list in indices_map.items():
            file_path = os.path.join(self.save_directory, f'{idx_type}_indices.txt')
            with open(file_path, 'w') as f:
                for subfolder_name, sample_idx in idx_list:
                    f.write(f"{subfolder_name}, {sample_idx}\n")


    def save_data_stats(self, data_stats):
        """
        Saves dataset statistics to a JSON file.

        Args:
            data_stats (dict): A dictionary containing dataset statistics.
        """
        data_stats_serializable = {
            key: [value[0].tolist(), value[1].tolist()]
            for key, value in data_stats.items()
            if key != "default" or len(data_stats) == 1
        }

        json_file_path = os.path.join(self.save_directory, 'data_stats.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(data_stats_serializable, json_file, indent=4)

        print(f"Data statistics saved to {json_file_path}")

    def save_hyperparameters(self):
        """
        Saves hyperparameters to an INI file.
        """
        config = configparser.ConfigParser()

        config.add_section('Model')
        for key, value in self.model_params.items():
            config.set('Model', key, str(value))
        
        config.add_section('Optimizer')
        for key, value in self.optimizer_params.items():
            config.set('Optimizer', key, str(value))

        config.add_section('Scheduler')
        for key, value in self.scheduler_params.items():
            config.set('Scheduler', key, str(value))

        config.add_section('Loss')
        for key, value in self.loss_params.items():
            config.set('Loss', key, str(value))

        config.add_section('Training')
        for key, value in self.training_params.items():
            config.set('Training', key, str(value))

        config.add_section('Data')
        for key, value in self.data.items():
            config.set('Data', key, str(value))
        
        if hasattr(self, 'augmentation_mapping') and self.augmentation_mapping:
            config.add_section('Data_augmentation')        
            for key, value in self.augmentation_mapping.items():
                config.set('Data_augmentation', key, str(value))
        
        ini_file_path = os.path.join(self.save_directory, 'hyperparameters.ini')
        with open(ini_file_path, 'w') as configfile:
            config.write(configfile)

        print(f"Hyperparameters saved to {ini_file_path}")

    def save_best_metrics(self, loss_dict, metrics_dict):
        """
        Saves validation loss and metrics history.

        Args:
            loss_dict (dict): Dictionary of loss values.
            metrics_dict (dict): Dictionary of metric values.
        """
        file_path = os.path.join(self.save_directory, "val_metrics_history.txt")

        with open(file_path, "w") as f:
            f.write("Validation Metrics History\n")
            f.write("=" * 30 + "\n\n")

            for epoch in sorted(loss_dict['val'].keys()):
                f.write(f"Epoch {epoch}:\n")
                f.write(f"  - Loss: {loss_dict['val'][epoch]:.4f}\n")
                for metric, values in metrics_dict['val'].items():
                    f.write(f"  - {metric}: {values[epoch - 1]:.4f}\n")
                f.write("\n" + "-" * 30 + "\n\n")

        print(f"Validation metrics history saved to {file_path}")

    def plot_learning_curves(self, loss_dict, metrics_dict):
        """
        Plots learning curves for loss and metrics over epochs.

        Args:
            loss_dict (dict): Dictionary of loss values.
            metrics_dict (dict): Dictionary of metric values.
        """
        epochs = list(loss_dict['train'].keys())
        train_loss_values = [loss_dict['train'][epoch] for epoch in epochs]
        val_loss_values = [loss_dict['val'][epoch] for epoch in epochs]

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        ax0 = axes[0]
        ax0.plot(epochs, train_loss_values, 'b-', label='Train Loss')
        ax0.plot(epochs, val_loss_values, 'r-', label='Val Loss')
        ax0.set_title('Loss')
        ax0.set_xlabel('Epochs')
        ax0.set_ylabel('Loss Value')
        ax0.legend()

        ax1 = axes[1]
        for metric in metrics_dict['train']:
            train_metric_values = [metrics_dict['train'][metric][epoch - 1] for epoch in epochs]
            val_metric_values = [metrics_dict['val'][metric][epoch - 1] for epoch in epochs]
            ax1.plot(epochs, train_metric_values, label=f'Train {metric}')
            ax1.plot(epochs, val_metric_values, label=f'Val {metric}')

        ax1.set_title('Metrics')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Metric Values')
        ax1.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_directory, 'learning_curves.png'), dpi=300)
        plt.close()
        print(f"Learning curves saved to {self.save_directory}/learning_curves.png")

    def save_confusion_matrix(self, conf_metric, model, val_dataloader, device):
        """
        Saves a confusion matrix plot.

        Args:
            conf_metric: Initialized confusion matrix metric instance.
            model: Trained model.
            val_dataloader: Validation dataset loader.
            device: Computation device.
        """
        model.eval()
        conf_metric.reset()

        final_preds, final_labels = [], []
        with torch.no_grad():
            for inputs, labels, _ in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                if self.num_classes > 1:
                    preds = torch.argmax(outputs, dim=1)
                else:
                    preds = (outputs > 0.5).to(torch.uint8)

                final_preds.append(preds.cpu())
                final_labels.append(labels.cpu())

        if final_preds and final_labels:
            preds_tensor = torch.cat(final_preds).to(device).long()
            labels_tensor = torch.cat(final_labels).to(device).long()

            if self.num_classes == 1:
                preds_tensor = preds_tensor.squeeze(1)
                labels_tensor = labels_tensor.squeeze(1)

            cm = conf_metric(preds_tensor, labels_tensor).cpu().numpy()
            cm_percent = (cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-6)) * 100

            plt.figure(figsize=(8, 6))
            plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues)
            plt.colorbar()

            tick_marks = [0, 1] if self.num_classes == 1 else list(range(self.num_classes))
            plt.xticks(tick_marks)
            plt.yticks(tick_marks)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")

            thresh = cm_percent.max() / 2.0
            for i in range(cm_percent.shape[0]):
                for j in range(cm_percent.shape[1]):
                    plt.text(j, i, f"{cm_percent[i, j]:.1f}%", horizontalalignment="center",
                             color="white" if cm_percent[i, j] > thresh else "black")

            plt.savefig(os.path.join(self.save_directory, "confusion_matrix.png"), dpi=300)
            plt.close()
            print(f"Confusion matrix saved to {self.save_directory}/confusion_matrix.png")