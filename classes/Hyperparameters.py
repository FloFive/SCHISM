import configparser  # Import configparser for .ini file handling


# Hyperparameters class
class Hyperparameters:
    def __init__(self, **kwargs):
        """
        Initializes the Hyperparameters instance with optional parameters.

        Parameters:
        img_side_length (int): The length of the image side (default is 512).
        num_sample (int): The number of samples to use (default is 1000).
        imgtype (str): The type of image ('scanner' or 'image', default is 'scanner').
        pretrained (bool): Whether to use a pretrained model (default is True).
        backbone (str): The backbone architecture to use (default is 'resnet50').
        image_preprocessing_functions (list): A list of functions for image preprocessing (default is empty).
        metrics (list): A list of metrics to evaluate the model (default is ["CategoricalCrossentropy, OneHotMeanIoU"]).
        featuremaps (int): The number of feature maps (default is 16).
        epochs (int): The number of training epochs (default is 20).
        val_split (float): The validation split ratio (default is 0.8).
        displaySummary (bool): Whether to display a summary of the model (default is True).
        maxNorm (int): The maximum norm for weight normalization (default is 3).
        learningRate (float): The learning rate for the optimizer (default is 1e-4).
        batchNorm (bool): Whether to use batch normalization (default is True).
        batch_size (int): The size of the batches for training (default is 3).
        save_model (bool): Whether to save the model after training (default is True).
        dropOut (bool): Whether to use dropout (default is True).
        dropOutRate (float): The dropout rate (default is 0.4).
        L2 (float): The L2 regularization parameter (default is 1e-4).
        early_stopping (list): A list of parameters for early stopping (default is empty).
        loss_early_stopping (bool): Whether to use loss for early stopping (default is False).
        patience (int): The number of epochs with no improvement after which training will be stopped (default is 25).
        run_name (str): The name of the current run (default is "default_run").
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
        """Returns a string representation of the Hyperparameters instance."""
        return f"Hyperparameters({self.__dict__})"

    def save_to_ini(self, file_path):
        """Saves hyperparameters to an .ini file.

        Parameters:
        file_path (str): The path to the .ini file where hyperparameters will be saved.
        """
        config = configparser.ConfigParser()
        config['Hyperparameters'] = {k: str(v) for k, v in self.__dict__.items()}
        with open(file_path, 'w') as configfile:
            config.write(configfile)

    @classmethod
    def load_from_ini(cls, file_path):
        """Loads hyperparameters from an .ini file.

        Parameters:
        file_path (str): The path to the .ini file from which hyperparameters will be loaded.

        Returns:
        Hyperparameters: An instance of the Hyperparameters class initialized with the loaded values.
        """
        config = configparser.ConfigParser()
        config.read(file_path)
        params = {k: v for k, v in config['Hyperparameters'].items()}
        return cls(**params)
